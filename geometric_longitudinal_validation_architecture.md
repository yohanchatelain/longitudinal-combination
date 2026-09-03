# Geometric Longitudinal Validation — Software Architecture

**Status:** Draft implementation architecture for `geometric_longitudinal_validation_plan.md`
**Date:** 2026-09-02

**Framing (revised):** the goal is not to build infrastructure that measures whether an
arbitrary frozen-random CNN happens to be invariant to nuisance transforms. The goal is to
**design a measurement architecture with nuisance-invariance as a first-class, engineered
property**, then use the confirmatory protocol to prove that design works — including
proving it works *better than* the unmodified architecture, since an engineered invariance
mechanism that doesn't measurably improve calibration/sham-suppression isn't earning its
complexity.

This reuses the reusable pieces already proven in `brainage_agg/validation/` (provenance,
metrics, gates, statistics scaffolding) and the cell-atomic SLURM orchestration pattern in
`brainage_agg/experiment/run_attribution_validation.py`, but the CNN measurement path
itself is now a three-layer designed pipeline, not a bare `_build_cnn` call.

## 0. What already exists vs. what's new

Reusable as-is: `validation/provenance.py`, `validation/metrics.py` (voxel AUROC/AUPRC —
covers Experiment 4), `agg/lme.py::make_lme_estimator`, `run_voxel_importance._build_cnn`
(the *raw*, non-invariant CNN factory — now the control arm, not the only arm),
`analysis/voxel_attribution.py`, `src/preprocessing.build_channels`, `src/io.load_cohort`,
cohort configs `brainage_agg/config.yaml` / `ppmi_config.yaml`.

Genuinely new:
- **Deformation generation with Jacobian-derived ground truth** (plan §3) — no prior code.
- **Three invariance mechanisms** (this document's core addition, §2 below) — no prior code.
- **Supervised CNN training** (plan §5.2) — confirmed absent; every CNN in this repo today
  is frozen-random (`_build_cnn` Xavier-inits then `freeze_model`s).
- **FastSurfer segmentation orchestration** (plan §5.3, amended 2026-09-02) — `recon-all`
  (cross-sectional and longitudinal) was ruled out entirely as too time-consuming; the
  comparator is now containerized FastSurfer (`deepmi/fastsurfer:cpu-latest` via
  Apptainer, `--seg_only` mode, `containers/fastsurfer-cpu.sif` pulled and verified
  working). No FreeSurfer license needed for seg-only mode. **Unblocked, not yet built.**
  FastSurfer has no longitudinal mode, so each timepoint is processed independently —
  see plan §5.3's explicit note that this is expected to be noisier on the zero-change
  calibration experiment (§6.1) than FreeSurfer's `-long` stream would have been.
- **Deformation-based morphometry baseline** — no registration library installed
  (SimpleITK/dipy/ANTsPy/ANTs CLI all absent). Still blocked.

## 1. Why the raw architecture isn't sufficient

The science plan's own motivation (§1) draws exactly the distinction this section acts on:

> An untrained CNN cannot learn dataset-specific artifacts through weight optimization. It
> can nevertheless remain sensitive to interpolation, registration, preprocessing,
> floating-point differences, random initialization, and architectural priors.

A frozen-random `CNN3D_DoubleConv`/`CNN3D_CovPool` has *no mechanism* that suppresses that
sensitivity — freezing prevents it from *learning* dataset artifacts, but does nothing
about the architecture's inherent response to sub-voxel interpolation jitter, non-deterministic
floating-point ops, or seed-dependent random-filter idiosyncrasies. Experiment 1 (zero-change
calibration) as originally scoped only *measures* whether that turns out to be a problem,
after the fact, with no path to fix it if it is one. This revision adds a designed fix and
tests it head-to-head against the unmodified architecture.

## 2. The three invariance mechanisms

Build order = dependency order = risk order (each is cheap to falsify before building the
next):

### 2.1 Deterministic preprocessing (`geometric/determinism.py`) — foundation

Nuisance sensitivity that comes from *non-reproducibility* (not from the architecture
itself) must be eliminated by construction, not modeled statistically:

- Baseline and follow-up flow through **byte-identical** preprocessing code
  (`build_channels`, resampling, scaling) — the only permitted difference between the two
  calls is the deformation field itself.
- Fixed floating-point precision end-to-end (float32 only, no mixed precision).
- `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, fixed thread counts for NumPy/SciPy ops in the resampling path.
- **Determinism self-test as a QC gate, run before any confirmatory cell**: pass the
  exact-duplicate condition through the full pipeline twice; require feature-vector output
  to match to a preregistered tight tolerance (near-bitwise, not just "small Cohen's d").
  This becomes gate 0, upstream of the existing Experiment 1 calibration gate — a bare
  reproducibility failure should never be allowed to masquerade as an architecture finding.

### 2.2 Band-limiting invariance layer (`geometric/invariance_layer.py`) — the core design

A fixed (non-learned) anti-alias filter inserted between preprocessing and the first conv
layer, applied identically to every input:

```
raw volume → deterministic preprocessing (2.1) → band-limit filter (this layer) →
build_channels → frozen/trained CNN
```

This directly targets the resampling-sham failure mode: sub-voxel interpolation jitter is
high-frequency by construction, so a filter with cutoff tied to the known interpolation
kernel support should suppress it while leaving a genuine multi-voxel deformation (2–10%
volume change over an anatomical region) largely intact. The filter has one free parameter
(cutoff/sigma) that must be **calibrated on the pilot split only** — same circularity
discipline already used for amplitude calibration in `validation/injection.py`:
`select_calibrated_amplitude` is the existing precedent to mirror, replacing "amplitude
nearest target Cohen's d" with "cutoff that minimizes sham response subject to retaining
≥X% of the pilot dose-response signal."

**This produces two CNN variants under test, not one**: `raw` (existing `_build_cnn`,
now the control arm) and `invariant` (raw + 2.1 + this filter). Every experiment in the
science plan (§6.1–6.5) runs both variants. The confirmatory claim is comparative: the
invariant variant must pass the calibration gate (§6.1) with a wider margin than raw
*and* must not lose meaningful dose-response separation (§6.2) doing it. If invariant
doesn't beat raw on calibration margin, the added architectural complexity isn't justified
and the plan's conclusion should say so, not quietly adopt it anyway.

### 2.3 Per-subject self-calibration (`geometric/self_calibration.py`) — measurement-level

Group-level calibration (Experiment 1, run once across all pilot/validation subjects) is
necessary but not sufficient — it says the *method* is calibrated on average, not that any
one subject's measurement is trustworthy. Add a per-subject null correction:

- For each subject, generate `k` (start at k=3–5, bounded to control compute cost per
  plan §7) independent null resamples from **that subject's own baseline scan only**
  (resampling-sham-style: same anatomy, independent interpolation/registration noise),
  through the identical deterministic pipeline (2.1) and invariance layer (2.2).
- Estimate that subject's own null mean/SD of feature-change magnitude from the `k` draws.
- Z-score (or robust-scale) the subject's observed feature-change against their own null
  distribution before it enters any group-level statistic (paired test, LME, bootstrap).
- **Anti-circularity constraint**: null resamples are computed blind to the subject's
  assigned condition/amplitude and derived only from the baseline image, so no deformation
  information leaks into the subject's own calibration.

This is what makes invariance a property of *each measurement*, not just an aggregate
statistic checked once at the end of the study — directly answering "design an architecture
that is invariant" rather than "measure whether one turned out to be."

## 3. Package layout

```
brainage_agg/geometric/
    __init__.py
    config.py                # locked-schema config loader (mirrors validation/config.py)
    deformation.py            # synthetic deformation fields + exact Jacobian-integral ground truth (plan §3)
    synthesis.py                # apply deformation to baseline -> synthetic follow-up, grid-identical
    qc.py                        # deformation QC gates (plan §3.4)
    determinism.py                 # §2.1 — deterministic preprocessing + reproducibility self-test (gate 0)
    invariance_layer.py             # §2.2 — band-limit filter, pilot-only cutoff calibration (raw vs invariant variants)
    self_calibration.py              # §2.3 — per-subject null-resample z-scoring
    frozen_cnn.py                     # adapter: paired feature-change through {raw, invariant} x {frozen, trained}
    trained_cnn.py                     # supervised training loop + checkpoint I/O (plan §5.2)
    decoder.py                          # ridge feature-change -> %volume-change, calibration subjects only (plan §6.3)
    fastsurfer.py                        # containerized FastSurfer --seg_only orchestration (plan §5.3, amended) — unblocked, not yet built
    morphometry_baseline.py               # registration/Jacobian reference method (plan §5.4) — blocked, see §6
    statistics.py                          # paired test, TOST equivalence, subject-clustered bootstrap, dose-response
    gates.py                                # gate 0 (determinism) + zero-change calibration + raw-vs-invariant comparison + §8 decision rule

brainage_agg/experiment/run_geometric_validation.py
brainage_agg/analysis/report_geometric_validation.py
brainage_agg/tests/test_geometric_validation.py

configs/geometric_longitudinal_validation.yaml   # separate run identifier, per plan §10
scripts/submit_geometric_validation.sh
slurm/geometric_validation_*.sbatch
```

Still architecturally separate from `brainage_agg/validation/*` / `attribution_validation.yaml`
per plan §10.

## 4. Data flow (per subject, per condition, per CNN variant ∈ {raw, invariant})

```
baseline image
      │
      ▼  deformation.py + qc.py → φ, ΔV_true (ground truth Jacobian integral)
      ▼  synthesis.py            → synthetic follow-up (grid-identical)
      │
      ├─ determinism.py: reproducibility self-test (gate 0) ───────────────┐
      ▼                                                                    │
 invariance_layer.py (variant=raw: identity; variant=invariant: band-limit)│
      ▼                                                                    │
 frozen_cnn.py / trained_cnn.py → paired feature-change                    │
      ▼                                                                    │
 self_calibration.py: z-score vs. subject's own null resamples ────────────┘
      ▼
 decoder.py (-> %vol, calibration subjects only)  /  statistics.py (dose-response, paired test)
      ▼
 gates.py: gate 0 → calibration gate (raw vs invariant, comparative) → §8 superiority rule
      ▼
 report_geometric_validation.py: Experiments 1-6, raw-vs-invariant comparison table
```

FastSurfer (independent per-timepoint segmentation, no longitudinal stream — plan §5.3
amendment) and the morphometry baseline sit alongside this CNN path, feeding the same
`statistics.py` / `gates.py`.

## 5. Config additions

`configs/geometric_longitudinal_validation.yaml` gains, beyond the schema already scoped:

```yaml
invariance:
  variants: [raw, invariant]        # both run through every experiment
  determinism:
    tolerance: <preregistered, near-bitwise>
  filter:
    kind: gaussian_lowpass           # or explicit band-limit kernel
    cutoff_candidates: [...]          # searched on pilot split only
    calibration_objective: minimize_sham_response_subject_to_dose_response_retention
  self_calibration:
    k_null_resamples: 5
    derived_from: baseline_only        # anti-circularity constraint, enforced in code

decision_gates:
  determinism: {tolerance: <as above>}
  invariant_vs_raw:
    invariant_must_not_reduce_dose_response_auroc_by_more_than: 0.05
    invariant_must_improve_calibration_margin: true   # else report "not justified", don't adopt silently
```

`geometric/config.py::validate_locked_config()` follows the exact pattern in
`validation/config.py` — reject drift from the preregistered variant list, filter search
grid, and `k_null_resamples`.

## 6. Open questions that still block Day 1

1. **RESOLVED (2026-09-02): FreeSurfer `recon-all` replaced by FastSurfer.** The user
   decided not to run `recon-all` at all — cross-sectional or longitudinal — regardless
   of hosting, on runtime-cost grounds (relayed via a peer session working the
   SimCLR-arm thread, and reflected in plan §5.3's amendment). FastSurfer (`--seg_only`,
   containerized via Apptainer, `containers/fastsurfer-cpu.sif`) is the replacement
   comparator, chosen over SynthSeg or dropping the arm entirely for its tie-in to this
   lab's own arXiv:2509.05238 FastSurfer numerical-variability paper. Consequence
   plan §1 now states explicitly: FastSurfer is itself a trained CNN-based tool, so it
   sits on the *learned* side of this study's classical-vs-learned distinction, not the
   classical side — the deformation-based morphometry baseline (§5.4/open question 2
   below) is the only remaining non-learned comparator. FastSurfer has no longitudinal
   mode, so each timepoint is processed independently; plan §5.3 documents the resulting
   noise-suppression gap against FreeSurfer's `-long` stream as an explicit limitation
   of the amendment, to be reported rather than absorbed silently. This package's
   `fastsurfer.py` (renamed from `freesurfer.py`) is unblocked but not yet built.
2. **Registration/Jacobian tool for the morphometry baseline** — no SimpleITK, dipy,
   ANTsPy, or `antsRegistration` binary found. Needs a decision (install ANTsPy vs. ANTs
   CLI module vs. defer this baseline as conditional follow-up).
3. **Deformation field implementation** — recommend a closed-form sum of localized radial
   basis bumps (pure NumPy/SciPy, analytically differentiable Jacobian, zero new
   dependency) over a B-spline FFD, since it avoids adding a registration-toolkit
   dependency this document already needs to resolve for the baseline method anyway.
4. **CNN training label/scope** (plan §5.2) — confirm whether disease/status-trained
   checkpoints are in scope for the initial 3–7 day stage or deferred.
5. **NEW — invariance-layer filter family**: is a simple isotropic Gaussian low-pass an
   acceptable first cut, or does the boundary-smoothness requirement in plan §3.4 argue
   for a filter shaped to the deformation taper (`taper_boundary_voxels`) instead of a
   generic cutoff? Affects how many candidate filters need pilot calibration.

## 7. Recommended build order

1. `geometric/config.py` + config file — nothing else testable without it.
2. `geometric/determinism.py` (§2.1) — cheapest to build, falsifies fastest: if the
   pipeline isn't reproducible, nothing downstream is trustworthy regardless of what else
   gets built.
3. `geometric/deformation.py` + `qc.py` + `synthesis.py` — pure NumPy/SciPy, unblocks the
   ground-truth side independent of any CNN work.
4. `geometric/invariance_layer.py` (§2.2) — the core new architecture piece; calibrate
   against `frozen_cnn.py` (raw variant already trivial via existing `_build_cnn`) to get
   an early end-to-end raw-vs-invariant comparison on the pilot split before touching
   FastSurfer or CNN training.
5. `geometric/self_calibration.py` (§2.3) — layers on top of 2–4 once the base
   raw-vs-invariant path is validated.
6. `geometric/trained_cnn.py` — biggest net-new engineering; both raw and invariant
   variants need a trained counterpart per plan §5.2.
7. `geometric/fastsurfer.py` — unblocked as of the plan §5.3 amendment (§6 item 1
   resolved); containerized `--seg_only` orchestration, no longer sequenced behind a
   blocker. `morphometry_baseline.py` remains blocked on §6 item 2.
8. `statistics.py` + `gates.py` + orchestrator + report — wire together once all
   method-side modules exist.
