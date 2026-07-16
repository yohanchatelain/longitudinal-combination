"""Per-CNN-feature importance weight vectors for voxel attribution.

Three weight-derivation methods, computed on the same per-subject aggregated
feature matrix ``X_agg`` (n_subjects, d_agg) and group labels ``bands_agg``
(produced by ``brainage_agg.analysis.group_diff.build_aggregated_vectors``):

  w_t      — Welch t-statistic (group_a vs group_b), signed.
             Thin wrapper around ``group_diff.run_option2_ttest``.
  w_shap   — signed mean TreeSHAP value for the group_a class, from a
             RandomForestClassifier fit on (X_agg, bands_agg).
  w_hybrid — sklearn permutation importance × sign(mean_a - mean_b), from
             the SAME fitted RandomForestClassifier as w_shap.

All three share the sign convention: positive ⇒ feature elevated in
``groups[0]`` (group_a), matching ``run_option2_ttest``'s ``t_stat``/``cohen_d``.

Standardization helpers (``rank_normalize``, ``zscore``) and cross-seed
``sign_stability_mask`` put the three vectors on equal footing before they
are backpropagated through the CNN (see ``voxel_attribution.backprop_from_w``).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from brainage_agg.analysis.group_diff import run_option2_ttest


# ---------------------------------------------------------------------------
# w_t — Welch t-statistic
# ---------------------------------------------------------------------------

def compute_w_t(
    X_agg: np.ndarray,
    bands_agg: np.ndarray,
    groups: tuple[str, str],
) -> np.ndarray:
    """Signed Welch t-statistic per feature (group_a vs group_b).

    Thin wrapper around ``group_diff.run_option2_ttest`` — positive values
    mean the feature is elevated in ``groups[0]``.
    """
    ttest_df = run_option2_ttest(X_agg, bands_agg, feature_names=None, groups=groups)
    return ttest_df["t_stat"].values.astype(np.float32)


# ---------------------------------------------------------------------------
# Shared RandomForest backing w_shap and w_hybrid
# ---------------------------------------------------------------------------

def fit_rf(
    X_agg: np.ndarray,
    bands_agg: np.ndarray,
    groups: tuple[str, str],
    seed: int,
    n_estimators: int = 200,
    min_samples_leaf: int = 5,
) -> RandomForestClassifier:
    """Fit a RandomForestClassifier distinguishing group_a (y=1) from group_b (y=0).

    One RF per (CNN seed, aggregation) backs both ``compute_w_shap`` and
    ``compute_w_hybrid`` — ``random_state=seed`` ties RF-fit randomness to the
    CNN seed (no separate RF-refit axis).

    ``min_samples_leaf=5`` (sklearn default is 1) prevents single-sample
    leaves: with n ~ 200-250 and d up to 23,040, an unconstrained RF reaches
    train_acc=1.0, which makes in-sample ``permutation_importance`` completely
    degenerate (``imp_nonzero_frac=0`` — observed on real data). Forcing
    leaves to cover >=5 samples leaves room for a permuted feature to change
    predictions, while ``oob_score`` stays informative.
    """
    group_a, _ = groups
    y = (bands_agg == group_a).astype(int)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        oob_score=True,
        n_jobs=1,
    )
    rf.fit(X_agg, y)
    return rf


# ---------------------------------------------------------------------------
# w_shap — signed mean TreeSHAP value for the group_a class
# ---------------------------------------------------------------------------

def compute_w_shap(
    rf: RandomForestClassifier,
    X_agg: np.ndarray,
    groups: tuple[str, str],
) -> np.ndarray:
    """Signed mean TreeSHAP value per feature for the group_a class.

    ``shap.TreeExplainer(rf).shap_values(X)`` return shape is version-
    dependent: shap 0.52.0 returns ``(n_samples, n_features, n_classes)``;
    older versions may return a list of per-class ``(n_samples, n_features)``
    arrays. Both are handled; the result is asserted to match
    ``X_agg.shape`` before averaging.
    """
    import shap

    class_a_idx = list(rf.classes_).index(1)  # y==1 == group_a, see fit_rf

    explainer = shap.TreeExplainer(rf)
    raw = explainer.shap_values(X_agg)

    if isinstance(raw, list):
        sv = np.asarray(raw[class_a_idx])
    else:
        raw = np.asarray(raw)
        sv = raw[:, :, class_a_idx] if raw.ndim == 3 else raw

    if sv.shape != X_agg.shape:
        raise ValueError(
            f"Unexpected SHAP value shape {sv.shape}; expected {X_agg.shape}. "
            f"shap.TreeExplainer return shape may have changed (raw type={type(raw)})."
        )

    return sv.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# w_hybrid — permutation importance x sign(mean_a - mean_b)
# ---------------------------------------------------------------------------

def compute_w_hybrid(
    rf: RandomForestClassifier,
    X_agg: np.ndarray,
    bands_agg: np.ndarray,
    groups: tuple[str, str],
    n_repeats: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Unsigned permutation importance, signed by group-mean difference.

    Returns ``(w_hybrid, diagnostics)``. ``diagnostics`` includes
    ``oob_score``, in-sample ``train_acc``, and ``imp_nonzero_frac`` —
    inspect these on real data before relying on ``w_hybrid``: with
    n_subjects ~ 50 and d_agg up to ~23,000, an overfit RF can make in-sample
    permutation importance degenerate (most importances ~ 0).
    """
    group_a, group_b = groups
    y = (bands_agg == group_a).astype(int)

    imp = permutation_importance(
        rf, X_agg, y, n_repeats=n_repeats, random_state=seed,
        scoring="accuracy", n_jobs=1,
    )
    sign = np.sign(
        X_agg[bands_agg == group_a].mean(axis=0) - X_agg[bands_agg == group_b].mean(axis=0)
    )
    w = (imp.importances_mean * sign).astype(np.float32)

    diagnostics = {
        "oob_score": float(rf.oob_score_),
        "train_acc": float(rf.score(X_agg, y)),
        "imp_nonzero_frac": float((imp.importances_mean != 0).mean()),
        "imp_mean": float(imp.importances_mean.mean()),
        "imp_std": float(imp.importances_mean.std()),
    }
    return w, diagnostics


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def rank_normalize(w: np.ndarray) -> np.ndarray:
    """Signed rank-normalization: ``sign(w) * rankit(|w|)``.

    Maps |w| to inverse-normal quantiles via average ranks, then reapplies
    the original sign. Zero-valued entries map to 0 regardless of rank.
    Applied identically to w_t, w_shap, and w_hybrid so all three share the
    same marginal distribution before backpropagation.
    """
    w = np.asarray(w, dtype=np.float64)
    sign = np.sign(w)
    ranks = rankdata(np.abs(w), method="average")
    quantiles = (ranks - 0.5) / len(w)
    out = sign * norm.ppf(quantiles)
    out[sign == 0] = 0.0
    return out.astype(np.float32)


def zscore(w: np.ndarray) -> np.ndarray:
    """Standard z-score: ``(w - mean(w)) / std(w)``.

    Used for the standalone "best single attribution" w_shap variant
    (magnitude-faithful), not for the three-way comparison.
    """
    w = np.asarray(w, dtype=np.float64)
    return ((w - w.mean()) / (w.std() + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-seed sign stability
# ---------------------------------------------------------------------------

def sign_stability_mask(w_seeds: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Flag features whose sign is unstable across CNN seeds.

    Args:
        w_seeds:   (n_seeds, d) raw (pre-standardization) weight vectors,
                   one row per CNN seed.
        threshold: a feature is flagged if more than this fraction of seeds
                   disagree with the majority sign (zero counts as a flip).

    Returns:
        (d,) bool mask, True = unstable. Apply to the *signed* standardized
        vector only (zero it out there); absolute maps keep these features.
    """
    w_seeds = np.asarray(w_seeds)
    signs = np.sign(w_seeds)  # (n_seeds, d) in {-1, 0, 1}
    n_seeds = signs.shape[0]
    majority = np.maximum((signs > 0).sum(axis=0), (signs < 0).sum(axis=0))
    frac_flip = 1.0 - majority / n_seeds
    return frac_flip > threshold
