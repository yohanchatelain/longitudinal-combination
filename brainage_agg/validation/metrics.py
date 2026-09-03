"""Localization, specificity, boundary, and concordance metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score


def localization_metrics(
    attribution: np.ndarray,
    injection_mask: np.ndarray,
    *,
    atlas: np.ndarray | None = None,
    left_labels: Iterable[int] = (),
    right_labels: Iterable[int] = (),
    top_k: int = 5,
) -> dict[str, float]:
    """Compute preregistered voxel and bilateral regional localization metrics."""
    score = np.abs(np.asarray(attribution, dtype=float))
    truth = np.asarray(injection_mask) > 0
    if score.shape != truth.shape:
        raise ValueError(f"Attribution/mask shape mismatch: {score.shape} != {truth.shape}")
    finite = np.isfinite(score)
    y_true = truth[finite].ravel().astype(np.uint8)
    y_score = score[finite].ravel()
    if y_true.min() == y_true.max():
        raise ValueError("Injection mask must contain both target and non-target voxels")
    prevalence = float(y_true.mean())
    auprc = float(average_precision_score(y_true, y_score))
    total_mass = float(y_score.sum())
    metrics: dict[str, float] = {
        "voxel_auroc": float(roc_auc_score(y_true, y_score)),
        "voxel_auprc": auprc,
        "voxel_prevalence": prevalence,
        "normalized_auprc_lift": auprc / prevalence,
        "attribution_mass_fraction": float(y_score[y_true.astype(bool)].sum() / total_mass) if total_mass > 0 else 0.0,
    }

    left = {int(x) for x in left_labels}
    right = {int(x) for x in right_labels}
    if atlas is not None and left and right:
        atlas_arr = np.asarray(atlas)
        if atlas_arr.shape != score.shape:
            raise ValueError("Atlas and attribution shapes differ")
        region_scores: dict[int, float] = {}
        for label in np.unique(atlas_arr):
            label = int(label)
            if label == 0:
                continue
            values = score[atlas_arr == label]
            region_scores[label] = float(values.mean()) if values.size else float("nan")
        order = sorted(region_scores, key=lambda label: (-region_scores[label], label))
        ranks = {label: rank for rank, label in enumerate(order, start=1)}
        left_rank = min((ranks.get(label, np.inf) for label in left), default=np.inf)
        right_rank = min((ranks.get(label, np.inf) for label in right), default=np.inf)
        metrics.update(
            {
                "left_target_rank": float(left_rank),
                "right_target_rank": float(right_rank),
                "bilateral_mean_rank": float(np.mean([left_rank, right_rank])),
                "bilateral_top5_hit": float(left_rank <= top_k and right_rank <= top_k),
            }
        )
    return metrics


def bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    *,
    statistic=np.mean,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Subject/seed bootstrap percentile interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(statistic(values))
    if values.size == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draws[index] = statistic(rng.choice(values, size=len(values), replace=True))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def map_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Pearson similarity between finite voxels in two maps."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Map shapes differ")
    keep = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    x, y = a[keep], b[keep]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def specificity_bootstrap(
    maps_by_target: Mapping[str, Sequence[np.ndarray]],
    prior_maps: Sequence[np.ndarray],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare within-target similarity against between-target and prior similarity.

    The resampling unit is seed. Each target must provide at least two seed maps;
    prior maps are paired to target maps by seed index.
    """
    targets = sorted(maps_by_target)
    if len(targets) < 2:
        raise ValueError("Specificity requires at least two targets")
    rng = np.random.default_rng(seed)
    rows = []
    for target in targets:
        maps = list(maps_by_target[target])
        if len(maps) < 2 or len(prior_maps) < len(maps):
            raise ValueError("Each target needs at least two maps and one paired prior map per seed")
        within = []
        between = []
        prior = []
        other_maps = [m for other in targets if other != target for m in maps_by_target[other]]
        for index, current in enumerate(maps):
            within.append(np.nanmean([map_similarity(current, m) for j, m in enumerate(maps) if j != index]))
            between.append(np.nanmean([map_similarity(current, m) for m in other_maps]))
            prior.append(map_similarity(current, prior_maps[index]))
        within = np.asarray(within)
        diff_between = within - np.asarray(between)
        diff_prior = within - np.asarray(prior)
        for comparison, differences in (("within_minus_between", diff_between), ("within_minus_prior", diff_prior)):
            draws = [float(np.nanmean(rng.choice(differences, size=len(differences), replace=True))) for _ in range(n_bootstrap)]
            rows.append(
                {
                    "target": target,
                    "comparison": comparison,
                    "estimate": float(np.nanmean(differences)),
                    "ci_lower": float(np.nanquantile(draws, 0.025)),
                    "ci_upper": float(np.nanquantile(draws, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def infer_tissue_class(region_name: str) -> str:
    """Map FreeSurfer region names to broad predeclared tissue classes."""
    name = region_name.lower()
    if name.startswith("ctx-"):
        return "cortical_gray"
    if "white-matter" in name or "wm-" in name or "corpus_callosum" in name:
        return "white_matter"
    if any(token in name for token in ("ventricle", "csf", "choroid")):
        return "csf"
    if any(token in name for token in ("putamen", "caudate", "pallidum", "thalamus", "hippocampus", "amygdala", "accumbens")):
        return "subcortical_gray"
    return "other"


def region_boundary_metrics(atlas: np.ndarray, lut: Mapping[int, str]) -> pd.DataFrame:
    """Quantify boundary fraction, log volume, hemisphere, and tissue class."""
    atlas = np.asarray(atlas)
    rows = []
    for label in np.unique(atlas):
        label = int(label)
        if label == 0:
            continue
        region = atlas == label
        eroded = binary_erosion(region, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
        n_voxels = int(region.sum())
        name = lut.get(label, f"label_{label}")
        lower = name.lower()
        hemisphere = "left" if lower.startswith(("left-", "ctx-lh", "wm-lh")) else "right" if lower.startswith(("right-", "ctx-rh", "wm-rh")) else "midline"
        rows.append(
            {
                "label_id": label,
                "region_name": name,
                "n_voxels": n_voxels,
                "log_volume": float(np.log(n_voxels)),
                "boundary_voxel_fraction": float((region & ~eroded).sum() / n_voxels),
                "hemisphere": hemisphere,
                "tissue_class": infer_tissue_class(name),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["region_volume_quintile"] = pd.qcut(frame["n_voxels"], 5, labels=False, duplicates="drop")
    return frame


def stratified_roi_permutation(
    x: np.ndarray,
    y: np.ndarray,
    strata: pd.DataFrame,
    *,
    n_permutations: int = 1999,
    seed: int = 0,
) -> dict[str, float]:
    """ROI concordance and a two-sided permutation p-value within all strata."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) != len(strata):
        raise ValueError("x, y, and strata must have equal lengths")
    keys = list(strata.astype(str).agg("|".join, axis=1))
    observed = map_similarity(x, y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    groups: dict[str, np.ndarray] = {}
    for key in sorted(set(keys)):
        groups[key] = np.flatnonzero(np.asarray(keys) == key)
    for permutation in range(n_permutations):
        permuted = y.copy()
        for indices in groups.values():
            permuted[indices] = rng.permutation(permuted[indices])
        null[permutation] = map_similarity(x, permuted)
    p = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (n_permutations + 1)
    return {"pearson_r": observed, "empirical_p_two_sided": float(p)}


def boundary_bias_regression(
    importance: pd.DataFrame,
    boundary: pd.DataFrame,
    *,
    score_column: str = "abs_importance",
) -> pd.DataFrame:
    """Regress regional importance on boundary fraction, log volume, and tissue.

    Coefficients are descriptive controls, not a named-region exclusion rule.
    Continuous predictors and the outcome are z-scored; tissue class is
    treatment-coded in sorted order. The returned table includes model R².
    """
    required_importance = {"label_id", score_column}
    required_boundary = {"label_id", "boundary_voxel_fraction", "log_volume", "tissue_class"}
    if missing := required_importance - set(importance):
        raise ValueError(f"Importance table missing columns: {sorted(missing)}")
    if missing := required_boundary - set(boundary):
        raise ValueError(f"Boundary table missing columns: {sorted(missing)}")
    frame = importance[["label_id", score_column]].merge(
        boundary[list(required_boundary)], on="label_id", how="inner"
    ).dropna()
    if len(frame) < 5:
        raise ValueError("Boundary regression requires at least five complete regions")
    y = frame[score_column].to_numpy(dtype=float)
    y_std = y.std(ddof=0)
    if y_std == 0:
        raise ValueError("Importance score is constant")
    y = (y - y.mean()) / y_std
    continuous = frame[["boundary_voxel_fraction", "log_volume"]].to_numpy(dtype=float)
    continuous_std = continuous.std(axis=0)
    if np.any(continuous_std == 0):
        raise ValueError("Boundary fraction and log volume must vary")
    continuous = (continuous - continuous.mean(axis=0)) / continuous_std
    tissue = pd.get_dummies(frame["tissue_class"], prefix="tissue", drop_first=True, dtype=float)
    names = ["intercept", "boundary_voxel_fraction", "log_volume", *tissue.columns.tolist()]
    design = np.column_stack([np.ones(len(frame)), continuous, tissue.to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    r_squared = 1.0 - float(np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    return pd.DataFrame(
        {
            "term": names,
            "estimate": coefficients,
            "r_squared": r_squared,
            "n_regions": len(frame),
        }
    )
