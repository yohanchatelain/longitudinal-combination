"""Per-feature LME slope estimation.

Two implementations selected by feature count:
- StatsmodelsLME: uses statsmodels.MixedLM for n_features <= threshold (e.g. FS ROI).
  Provides proper BLUP slopes with variance-component shrinkage.
- VectorizedOLSSlopes: vectorized per-subject OLS across all features simultaneously.
  Used for high-dimensional CNN features where per-feature MixedLM would be prohibitively
  slow. Documented fallback per plan §3.

Both expose the same fit/transform API.
"""
from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _center_ages(ages: np.ndarray) -> tuple[np.ndarray, float]:
    """Return globally centered ages and the mean used."""
    mu = float(ages.mean())
    return ages - mu, mu


def _per_subject_ols_slopes(
    X: np.ndarray,
    ages_c: np.ndarray,
    subject_ids: np.ndarray,
    unique_subjects: np.ndarray,
) -> np.ndarray:
    """Compute per-subject OLS slope for each feature.

    Returns slopes (n_subjects × n_features). Subjects with only one timepoint
    or zero age variance get a zero slope.
    """
    n_feat = X.shape[1]
    slopes = np.zeros((len(unique_subjects), n_feat), dtype=np.float64)
    for i, sid in enumerate(unique_subjects):
        mask = subject_ids == sid
        a = ages_c[mask]
        x = X[mask]
        denom = float((a ** 2).sum())
        if denom > 1e-12:
            slopes[i] = (a[:, None] * x).sum(axis=0) / denom
    return slopes


# ---------------------------------------------------------------------------
# VectorizedOLSSlopes (used for CNN, or when statsmodels is unavailable)
# ---------------------------------------------------------------------------

class VectorizedOLSSlopes:
    """Fast vectorized per-subject OLS slope estimator (documented fallback).

    Shrinkage toward the population slope using empirical Bayes:
        b_s* = w_s * b_s^OLS + (1 - w_s) * beta_pop
        w_s  = sigma2_b / (sigma2_b + sigma2_eps / n_s)
    where sigma2_b and sigma2_eps are estimated from the training set.
    """

    def __init__(self) -> None:
        self.beta_pop_: np.ndarray | None = None       # (n_features,)
        self.sigma2_b_: np.ndarray | None = None       # (n_features,) between-subject slope var
        self.sigma2_eps_: np.ndarray | None = None     # (n_features,) residual var
        self.age_mean_: float = 0.0
        self.train_subjects_: np.ndarray | None = None
        self.train_slopes_: np.ndarray | None = None   # (n_train_subjects, n_features)

    def fit(
        self,
        X: np.ndarray,
        ages: np.ndarray,
        subject_ids: np.ndarray,
    ) -> "VectorizedOLSSlopes":
        ages_c, self.age_mean_ = _center_ages(ages)
        unique_subjects = np.unique(subject_ids)

        # Population slope: pooled OLS across all training samples
        denom_pop = float((ages_c ** 2).sum())
        if denom_pop > 1e-12:
            self.beta_pop_ = (ages_c[:, None] * X).sum(axis=0) / denom_pop
        else:
            self.beta_pop_ = np.zeros(X.shape[1], dtype=np.float64)

        # Per-subject OLS slopes
        raw_slopes = _per_subject_ols_slopes(X, ages_c, subject_ids, unique_subjects)

        # Estimate between-subject slope variance and residual variance
        self.sigma2_b_ = np.var(raw_slopes, axis=0, ddof=1).clip(1e-12)

        # Residual variance: pooled over all subjects
        residuals_sq_sum = np.zeros(X.shape[1], dtype=np.float64)
        n_total = 0
        for i, sid in enumerate(unique_subjects):
            mask = subject_ids == sid
            a = ages_c[mask]
            predicted = a[:, None] * raw_slopes[i]
            res = X[mask] - predicted
            residuals_sq_sum += (res ** 2).sum(axis=0)
            n_total += mask.sum()
        self.sigma2_eps_ = (residuals_sq_sum / max(n_total - 2, 1)).clip(1e-12)

        self.train_subjects_ = unique_subjects
        self.train_slopes_ = raw_slopes
        return self

    def _shrink(
        self,
        raw_slopes: np.ndarray,
        n_per_subject: np.ndarray,
    ) -> np.ndarray:
        """Apply empirical Bayes shrinkage toward population slope."""
        # w_s = sigma2_b / (sigma2_b + sigma2_eps / n_s)  — broadcast over features
        n_arr = n_per_subject[:, None]  # (n_subjects, 1)
        w = self.sigma2_b_ / (self.sigma2_b_ + self.sigma2_eps_ / n_arr)
        # BLUP slope = beta_pop + w * (raw_slope - beta_pop)
        return self.beta_pop_ + w * (raw_slopes - self.beta_pop_)

    def transform(
        self,
        X: np.ndarray,
        ages: np.ndarray,
        subject_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (subjects, slopes) arrays.

        Returns:
            unique_subjects: (n_subjects,) subject ID array
            slopes:          (n_subjects, n_features) BLUP slope matrix
        """
        ages_c = ages - self.age_mean_
        unique_subjects = np.unique(subject_ids)
        raw_slopes = _per_subject_ols_slopes(X, ages_c, subject_ids, unique_subjects)
        n_per_subject = np.array([(subject_ids == s).sum() for s in unique_subjects])
        slopes = self._shrink(raw_slopes, n_per_subject)
        return unique_subjects, slopes


# ---------------------------------------------------------------------------
# StatsmodelsLME (used for FS ROI, n_features <= threshold)
# ---------------------------------------------------------------------------

class StatsmodelsLME:
    """Per-feature LME using statsmodels.MixedLM.

    Model: feature_j ~ 1 + age + (1 + age | subject)
    Extracts BLUP (random) slope per subject = fixed slope + random deviation.

    For test subjects (not seen during fit), applies trained fixed effects and
    computes empirical BLUP from their own timepoints.
    """

    def __init__(self) -> None:
        self.age_mean_: float = 0.0
        self.beta_pop_: np.ndarray | None = None       # fixed slope (n_features,)
        self.random_effects_: dict | None = None       # {subject_id: slope_deviation (n_features,)}
        self.sigma2_b_: np.ndarray | None = None
        self.sigma2_eps_: np.ndarray | None = None
        self.train_subjects_: set = set()

    def fit(
        self,
        X: np.ndarray,
        ages: np.ndarray,
        subject_ids: np.ndarray,
    ) -> "StatsmodelsLME":
        import statsmodels.formula.api as smf
        import pandas as pd

        ages_c, self.age_mean_ = _center_ages(ages)
        n_feat = X.shape[1]

        beta_pop = np.zeros(n_feat, dtype=np.float64)
        random_effects: dict[str, np.ndarray] = {}
        sigma2_b = np.zeros(n_feat, dtype=np.float64)
        sigma2_eps = np.zeros(n_feat, dtype=np.float64)

        for j in range(n_feat):
            df = pd.DataFrame(
                {"y": X[:, j], "age": ages_c, "subject": subject_ids}
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(
                        "y ~ age",
                        df,
                        groups=df["subject"],
                        re_formula="~age",
                    ).fit(reml=True, method="lbfgs", disp=False)

                beta_pop[j] = float(model.fe_params.get("age", 0.0))
                # random slope deviation per subject
                for sid, re_dict in model.random_effects.items():
                    dev = float(re_dict.get("age", 0.0))
                    if sid not in random_effects:
                        random_effects[sid] = np.zeros(n_feat, dtype=np.float64)
                    random_effects[sid][j] = dev

                # variance components (for BLUP of unseen subjects)
                vc = model.cov_re
                sigma2_b[j] = float(vc.get("age", vc.iloc[-1, -1])) if hasattr(vc, "get") else float(vc[-1, -1])
                sigma2_eps[j] = float(model.scale)

            except Exception:
                # Fallback for degenerate features: use pooled OLS slope
                denom = float((ages_c ** 2).sum())
                beta_pop[j] = float((ages_c * X[:, j]).sum() / denom) if denom > 1e-12 else 0.0

        self.beta_pop_ = beta_pop
        self.random_effects_ = random_effects
        self.sigma2_b_ = sigma2_b.clip(1e-12)
        self.sigma2_eps_ = sigma2_eps.clip(1e-12)
        self.train_subjects_ = set(np.unique(subject_ids))
        return self

    def transform(
        self,
        X: np.ndarray,
        ages: np.ndarray,
        subject_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (subjects, BLUP_slopes) arrays.

        Train subjects use fitted random effects directly.
        Test subjects get empirical BLUP computed from stored variance components.
        """
        ages_c = ages - self.age_mean_
        unique_subjects = np.unique(subject_ids)
        n_feat = X.shape[1]
        slopes = np.zeros((len(unique_subjects), n_feat), dtype=np.float64)

        for i, sid in enumerate(unique_subjects):
            mask = subject_ids == sid
            if sid in self.train_subjects_ and self.random_effects_ is not None and str(sid) in self.random_effects_:
                # Trained subject: use stored BLUP
                slopes[i] = self.beta_pop_ + self.random_effects_[str(sid)]
            else:
                # Unseen subject: empirical BLUP
                a = ages_c[mask]
                x = X[mask]
                n_s = len(a)
                denom = float((a ** 2).sum())
                if denom > 1e-12:
                    ols_dev = (a[:, None] * (x - a[:, None] * self.beta_pop_)).sum(axis=0) / denom
                else:
                    ols_dev = np.zeros(n_feat, dtype=np.float64)
                # Shrinkage: w = sigma2_b / (sigma2_b + sigma2_eps / n_s)
                w = self.sigma2_b_ / (self.sigma2_b_ + self.sigma2_eps_ / max(n_s, 1))
                slopes[i] = self.beta_pop_ + w * ols_dev

        return unique_subjects, slopes


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_lme_estimator(n_features: int, threshold: int = 700):
    """Return the appropriate LME estimator for the given feature count."""
    if n_features <= threshold:
        try:
            import statsmodels  # noqa: F401
            return StatsmodelsLME()
        except ImportError:
            pass
    return VectorizedOLSSlopes()
