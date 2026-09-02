"""SimCLR trained-vs-untrained comparator: Phase 0a (Gate 0) of the SimCLR arm.

See simclr_experimental_plan.md for the science, execution_plan.md for how this fits
into the broader confirmatory study. This package deliberately does not import anything
from brainage_agg/validation or brainage_agg/geometric — this worktree's branch predates
both, and this package is scoped to run independently of them.
"""

from .gate0 import (
    Gate0Result,
    activation_statistics,
    linear_probe_check,
    recompute_batchnorm_running_stats,
    run_gate0,
)
from .model import load_simclr_backbone

__all__ = [
    "Gate0Result",
    "activation_statistics",
    "linear_probe_check",
    "load_simclr_backbone",
    "recompute_batchnorm_running_stats",
    "run_gate0",
]
