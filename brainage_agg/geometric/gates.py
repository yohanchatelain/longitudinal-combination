"""Decision gates for the geometric longitudinal validation study.

Plan §8's six numbered superiority criteria, its zero-change calibration
precondition (§6.1), and the architecture §2.2 raw-vs-invariant comparison gate.
Each function checks ONE named criterion against caller-supplied, already-computed
inputs (typically from `statistics.py`); `evaluate_superiority_claim` composes them
exactly per plan §8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: dict[str, bool]
    values: dict[str, float]


def check_calibration_gate(
    false_positive_rate: float,
    upper_ci_bound: float,
    *,
    fpr_max: float = 0.05,
    upper_ci_max: float = 0.10,
) -> GateResult:
    """Plan §6.1 zero-change calibration gate: "A method is eligible for comparison
    only after passing zero-change calibration."
    """
    fpr_pass = false_positive_rate <= fpr_max
    ci_pass = upper_ci_bound <= upper_ci_max
    return GateResult(
        passed=fpr_pass and ci_pass,
        reasons={"false_positive_rate_ok": fpr_pass, "upper_ci_bound_ok": ci_pass},
        values={"false_positive_rate": float(false_positive_rate), "upper_ci_bound": float(upper_ci_bound)},
    )


def check_error_difference_favors_method(paired_error_difference: float) -> bool:
    """Plan §8 rule 1: "the paired error difference favors the untrained CNN."

    `paired_error_difference` is comparator_error - method_error, computed by the
    caller; positive means the method under test has the lower error.
    """
    return paired_error_difference > 0.0


def check_bootstrap_excludes_zero(bootstrap_ci: dict) -> bool:
    """Plan §8 rule 2: "the subject-clustered 95% bootstrap interval excludes zero."

    Takes a `statistics.subject_clustered_bootstrap_ci` result directly.
    """
    return bool(bootstrap_ci["excludes_zero"])


def check_practically_meaningful_improvement(
    relative_improvement_pct: float,
    *,
    min_improvement_pct: float = 20.0,
) -> bool:
    """Plan §8 rule 3: "the improvement is practically meaningful, such as at least a
    20% reduction in magnitude or localization error."
    """
    return relative_improvement_pct >= min_improvement_pct


def check_false_positive_rate_no_worse(
    method_fpr: float,
    comparator_fpr: float,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Plan §8 rule 4: "the false-positive rate is no worse.\""""
    return method_fpr <= comparator_fpr + tolerance


def check_consistent_direction_across_cohorts(direction_by_cohort: dict[str, float]) -> bool:
    """Plan §8 rule 5: "the direction is consistent in NKI and PPMI."

    `direction_by_cohort` maps cohort name to a signed effect (e.g. an error
    difference); consistent means every cohort's sign agrees -- all positive or all
    negative, never mixed, and never exactly zero (an undecided direction doesn't
    count as consistent).
    """
    if len(direction_by_cohort) < 2:
        raise ValueError("check_consistent_direction_across_cohorts requires at least two cohorts")
    signs = {float(np.sign(value)) for value in direction_by_cohort.values()}
    return signs in ({1.0}, {-1.0})


def check_not_single_seed_driven(
    per_seed_effect: np.ndarray,
    *,
    max_single_seed_share: float = 0.5,
) -> bool:
    """Plan §8 rule 6: "the result is not driven by one favorable random seed."

    Fails if the single largest-magnitude seed accounts for more than
    `max_single_seed_share` of the total across-seed effect magnitude, or if
    dropping that seed flips the sign of the mean effect.
    """
    per_seed_effect = np.asarray(per_seed_effect, dtype=float)
    if len(per_seed_effect) < 2:
        raise ValueError("check_not_single_seed_driven requires at least two seeds")
    total_magnitude = float(np.sum(np.abs(per_seed_effect)))
    if total_magnitude == 0:
        return True
    dominant_index = int(np.argmax(np.abs(per_seed_effect)))
    dominant_share = float(np.abs(per_seed_effect[dominant_index])) / total_magnitude
    if dominant_share > max_single_seed_share:
        return False
    remaining_mean = float(np.delete(per_seed_effect, dominant_index).mean())
    full_mean = float(per_seed_effect.mean())
    return bool(np.sign(remaining_mean) == np.sign(full_mean))


@dataclass(frozen=True)
class SuperiorityClaimResult:
    superior: bool
    checks: dict[str, bool]


def evaluate_superiority_claim(
    *,
    paired_error_difference: float,
    bootstrap_ci: dict,
    relative_improvement_pct: float,
    method_fpr: float,
    comparator_fpr: float,
    direction_by_cohort: dict[str, float],
    per_seed_effect: np.ndarray,
    min_improvement_pct: float = 20.0,
    max_single_seed_share: float = 0.5,
) -> SuperiorityClaimResult:
    """Plan §8: a method is superior to a comparator only if ALL six numbered
    criteria hold -- this applies equally to "untrained CNN vs. comparator" and, per
    architecture §2.2, "invariant CNN variant vs. raw CNN variant."
    """
    checks = {
        "error_difference_favors_method": check_error_difference_favors_method(paired_error_difference),
        "bootstrap_excludes_zero": check_bootstrap_excludes_zero(bootstrap_ci),
        "practically_meaningful": check_practically_meaningful_improvement(
            relative_improvement_pct, min_improvement_pct=min_improvement_pct,
        ),
        "false_positive_rate_no_worse": check_false_positive_rate_no_worse(method_fpr, comparator_fpr),
        "consistent_direction_across_cohorts": check_consistent_direction_across_cohorts(direction_by_cohort),
        "not_single_seed_driven": check_not_single_seed_driven(
            per_seed_effect, max_single_seed_share=max_single_seed_share,
        ),
    }
    return SuperiorityClaimResult(superior=all(checks.values()), checks=checks)


def check_invariant_variant_justified(
    *,
    invariant_dose_response_auroc: float,
    raw_dose_response_auroc: float,
    invariant_calibration_margin: float,
    raw_calibration_margin: float,
    max_auroc_reduction: float = 0.05,
) -> GateResult:
    """Architecture §2.2: the invariant CNN variant earns its added complexity only
    if it (a) does not lose more than `max_auroc_reduction` of dose-response AUROC
    relative to raw, and (b) improves calibration margin over raw. If either fails,
    the conclusion should say the invariance layer is "not justified" for this
    cell, not adopt it anyway.

    `calibration_margin` is caller-defined (e.g. `upper_ci_max` minus the observed
    false-positive upper CI bound); larger is safer.
    """
    auroc_drop = raw_dose_response_auroc - invariant_dose_response_auroc
    dose_response_retained = auroc_drop <= max_auroc_reduction
    calibration_margin_improved = invariant_calibration_margin > raw_calibration_margin
    return GateResult(
        passed=dose_response_retained and calibration_margin_improved,
        reasons={
            "dose_response_retained": dose_response_retained,
            "calibration_margin_improved": calibration_margin_improved,
        },
        values={
            "auroc_drop": float(auroc_drop),
            "invariant_calibration_margin": float(invariant_calibration_margin),
            "raw_calibration_margin": float(raw_calibration_margin),
        },
    )
