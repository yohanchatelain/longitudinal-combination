# Longitudinal CNN Feature Benchmark

Benchmarks untrained, randomly-initialized 3D CNNs against FreeSurfer ROI features for predicting brain age and detecting group differences in longitudinal MRI cohorts (NKI and PPMI). The core interpretability tool is **voxel importance**: a gradient-based method that maps which brain regions drive CNN predictions.

---

## Table of Contents

1. [Overview](#overview)
2. [Method: Voxel Importance via Gradient Attribution](#method-voxel-importance-via-gradient-attribution)
3. [Quick Start: Voxel Importance API](#quick-start-voxel-importance-api)
4. [Interactive Dashboard](#interactive-dashboard)
5. [Full Pipeline](#full-pipeline)
6. [Data Assumptions](#data-assumptions)
7. [Leakage Control](#leakage-control)
8. [Install](#install)
9. [Registration-Free Fixed Feature Plan](#registration-free-fixed-feature-plan)

---

## Overview

The project answers two questions:

- **Which aggregation of longitudinal features works best for brain age prediction?**
  Five strategies are benchmarked: `mean`, `concatenation`, `annualized_rate`, `lme_slope`, and `difference`.

- **Which brain regions drive CNN predictions — and do they match known anatomy?**
  Gradient-based voxel importance maps CNN feature weights back to voxel space and projects them onto the FreeSurfer Desikan-Killiany atlas.

Two cohorts are supported:

| Cohort | Groups | Task |
|--------|--------|------|
| NKI (Rockland) | Child vs Adult | Brain age prediction |
| PPMI | PD vs Healthy Control | Parkinson's disease classification |

> **Interpretation status.** The existing 36-cell voxel-attribution sweep is
> exploratory. Its regional maps must not be interpreted as biological
> localization unless the preregistered positive-control, maxT calibration,
> and target-specificity gates in `configs/attribution_validation.yaml` pass.

---

## Registration-Free Fixed Feature Plan

This extension will test whether a deterministic physical-space representation
can replace registration-dependent features. The representation has no weights
fitted to this project's images or outcomes; only the downstream prediction
model is trained. The confirmatory question is whether it retains useful brain
morphology while reducing sensitivity to acquisition and pose nuisances.

### Operational definition and data boundary

- The primary input is the original raw T1-weighted image and its complete
  NIfTI/MGZ affine. FreeSurfer `brain.mgz`, `brainmask.mgz`, and parcellations are
  not valid primary inputs because they have already been conformed and inherit
  FreeSurfer preprocessing. They remain registered comparison baselines only.
- "Registration-free" means no atlas alignment, inter-visit alignment,
  optimized affine/deformation field, or learned representation encoder.
  Lossless array-axis permutation and flipping to canonical RAS+ orientation is
  allowed and recorded; it does not interpolate voxel values or align anatomy.
- A frozen third-party brain-extraction procedure may be used as preprocessing,
  but it must be selected before confirmatory evaluation, must not be trained or
  fine-tuned on either cohort, and must be identified by model/tool hash. Its
  contribution is evaluated separately from the fixed representation.
- The target is stability to acquisition resolution, added background,
  non-cropping pose changes, noise, bias field, and monotone intensity changes.
  The method does **not** claim invariance to missing anatomy or anatomical
  scale: brain size and genuine atrophy are signals that should remain visible.

The raw-image manifest must include subject and visit identifiers, image path,
input checksum, affine, voxel spacing, orientation, dimensions, scanner/site
when available, and elapsed time. All visits from a subject must enter the same
cross-validation fold.

### Native-space preprocessing and quality control

Each visit is processed independently in this fixed order:

1. Validate the affine, finite intensities, supported dimensionality, and full
   brain coverage. Reject truncated brains rather than treating missing tissue
   as a nuisance transformation.
2. Reorient to canonical RAS+ with axis permutation/flipping only. Reject or
   separately flag images whose affine contains unsupported shear or invalid
   physical spacing.
3. Apply a pinned N4 bias-field implementation with fixed parameters.
4. Generate the brain mask with the locked external procedure. Apply fixed
   topology, minimum-volume, boundary-contact, and left/right coverage QC;
   never fall back silently to `image > 0`.
5. Normalize intensities within the mask with a locked robust rule, including
   fixed clipping percentiles and explicit handling of constant or nearly
   constant images. Set background to zero.
6. Retain the native physical grid. Do not conform or resize the primary input
   to a template shape.

Save the corrected image and mask checksums, QC measurements, voxel-to-world
affine, physical field of view, mask volume in mm³, software versions, and any
warning or rejection reason. A manually reviewed stratified subset will measure
mask failure rates by cohort, site, and diagnostic group before prediction is
evaluated.

### Locked fixed representation

The primary extractor is one pinned implementation of a 3D solid-harmonic
scattering transform. "Equivalent wavelet bank" is not a confirmatory option.
The configuration locks:

- filter equations and normalization;
- orders 0, 1, and 2 only;
- physical radial scales in millimetres;
- angular orders and orientation-energy pooling;
- averaging scale, padding, boundary correction, and mask erosion rules;
- numerical precision, backend version, and deterministic execution settings;
- coefficient names, ordering, and invalid-value policy.

Filters are discretized from the voxel-to-world metric, not from voxel indices
alone, so anisotropic spacing is represented in physical units. A scan is
rejected if its spacing or affine is outside the preregistered supported range.
Unit tests on analytic phantoms must show that every supported grid produces the
same coefficient names and acceptably close values before cohort extraction.

Orientation pooling and spatial averaging are expected to improve stability;
they do not imply exact invariance. Likewise, multiple physical filter scales
address acquisition sampling but do not normalize anatomical size. These are
empirical properties assessed by the locked robustness tests below.

### Registration-free spatial summaries

The primary feature vector contains, for every scattering coefficient:

1. a whole-mask summary; and
2. summaries over four concentric intrinsic radial shells defined by physical
   distance from the mask centroid, normalized only for shell assignment.

Radial shells retain coarse center-to-periphery information without imposing
voxelwise or axis-aligned regional correspondence. Their continuous-space
definition is rotation-invariant; discretization error is measured in the
phantom and perturbation tests. Mask volume, rotation-invariant principal-axis
lengths, and intensity quantiles are separate named covariates; they are never
mixed into the scattering coefficients.

Axis-dependent grids, octants, and hemispheric summaries are excluded from the
primary rotation-robust representation. A preregistered secondary analysis may
add left/right summaries only for scans passing orientation QC, and must label
the resulting feature family as orientation-dependent.

### Longitudinal representation

Extract one visit-level vector independently for every scan. For subjects with
two visits, compare these non-redundant, preregistered candidates:

- baseline state \(f(t_0)\);
- endpoint mean plus annualized change
  \([(f(t_0)+f(t_1))/2,\; (f(t_1)-f(t_0))/\Delta t]\); and
- annualized change alone.

Do not concatenate \(f(t_0)\), \(f(t_1)\), and their exact difference in the
same design matrix. For three or more visits, the longitudinal candidate is the
per-subject intercept at a fixed reference time plus a per-feature ordinary
least-squares slope. Minimum/maximum elapsed-time rules and the treatment of
irregular visits are locked in configuration before extraction.

Representation choice is a model hyperparameter selected only within inner
training folds. Elapsed time and number of visits are retained as explicit
covariates and reported for every outer fold.

### Common leakage-safe prediction protocol

All feature families use one shared evaluator and identical saved outer folds:

- subject-grouped nested cross-validation for NKI regression and PPMI
  classification;
- outcome-aware stratification where feasible, with a predefined fallback for
  small strata;
- fold-local feature scaling, near-zero-variance removal, optional feature
  selection/dimensionality reduction, longitudinal-representation selection,
  and hyperparameter selection;
- a primary elastic-net linear/logistic model; RBF SVM and gradient-boosted
  trees only as prespecified secondary models; and
- paired fold-level comparisons with confidence intervals, not independent
  comparisons of separately randomized splits.

The same protocol evaluates the fixed representation, volume/intensity-only
native-space features, FreeSurfer ROI features, and the existing frozen random
CNN features. No outer-fold result may influence the extractor, perturbation
settings, model search space, or acceptance criteria.

### Robustness and signal-preservation validation

Before confirmatory runs, a versioned configuration locks perturbation levels,
random seeds, primary metrics, non-inferiority/superiority margins, and failure
criteria. Development pilots may set those values, but pilot subjects and
results are excluded from the confirmatory assessment.

On held-out outer-fold subjects, apply deterministic perturbation sets for:

- rigid rotations and translations with padding and verified complete brain
  coverage;
- resolution changes implemented by a documented downsample/reconstruction
  operator;
- added or removed background that never intersects the brain mask;
- noise, smooth multiplicative bias fields, and monotone intensity transforms;
- simulated focal volume loss at multiple locked doses; and
- real test-retest scans when available.

Run perturbations in two modes: first reuse the original mask to isolate the
representation, then rerun N4 and brain extraction to measure the complete
pipeline. Report coefficient-wise reliability, feature-vector cosine and
standardized L2 change, prediction change/class flips, test-retest ICC, runtime,
memory, QC rejection rate, and failures. Signal preservation is assessed by a
monotone dose-response to simulated atrophy and by sensitivity to real
longitudinal change. Robustness is accepted only if the preregistered prediction
and feature-stability margins pass without failing the signal-preservation gate.

### Reproducibility and implementation sequence

Every artifact records the raw input and mask checksums, resolved configuration
and its hash, feature schema version, dependency/container versions, thread and
device settings, source commit, QC results, and fold assignment. Feature files
are immutable and are written to a new versioned output directory.

Implementation proceeds through explicit gates:

1. add the raw-image manifest and preprocessing/QC command;
2. implement the physical-space extractor and phantom invariance tests;
3. define and validate the immutable feature schema;
4. add the shared nested-CV regression/classification evaluator and saved paired
   folds;
5. run development-only feasibility and resource pilots;
6. freeze the confirmatory configuration and acceptance margins; and
7. run extraction, prediction, robustness tests, and a completeness/hash audit.

The confirmatory campaign starts only after the raw-data boundary, mask audit,
phantom tests, paired evaluator, and configuration freeze have all passed.

---

## Method: Voxel Importance via Gradient Attribution

### Why?

CNNs produce high-dimensional feature vectors (7680 dims for `CNN3D_DoubleConv`) with no built-in spatial interpretation. Voxel importance answers: *which voxels of the input brain scan were most responsible for a group difference in the CNN features?*

### How it works

Given a **frozen, randomly-initialized** CNN and a brain scan `x`:

1. **Compute group-difference weights** `w` — run a Welch t-test between two groups (e.g. Child vs Adult) on the CNN feature matrix. The t-statistic `wᵢ` for feature `i` measures how strongly that feature separates the groups.

2. **Form a scalar proxy** for the group difference:

   ```
   S = Σᵢ  wᵢ · CNNᵢ(x)
   ```

   `S` is large when the features of `x` look like the group with higher mean features.

3. **Backpropagate** to get `∂S/∂x` — the gradient with respect to each input voxel. Voxels with large `|∂S/∂x|` strongly influence the group separation.

4. **Memory optimization**: the input is trilinearly downsampled by 2× (256³ → 128³) before the forward pass. This reduces first-layer activation memory from ~4.3 GB to ~0.54 GB, fitting on a 14 GB GPU without gradient checkpointing. The gradient is upsampled back to the original resolution.

5. **Project to atlas**: average the gradient map within each brain region defined by the FreeSurfer aparc+aseg parcellation (Desikan-Killiany, ~90 regions). Uses `np.bincount` for an O(n_voxels) single pass — ~150× faster than iterating over labels.

### Signed vs absolute maps

| Map | Meaning |
|-----|---------|
| `signed_map` | Positive voxels push features toward group A; negative voxels push toward group B. Useful for directionality. |
| `abs_map` | Total spatial sensitivity — which regions matter, regardless of direction. |

---

## Quick Start: Voxel Importance API

```python
import nibabel as nib
import numpy as np
from src.models import CNN3D_DoubleConv, freeze_model
from brainage_agg.analysis.voxel_attribution import compute_voxel_importance

# 1. Build (or load) a frozen CNN
model = freeze_model(CNN3D_DoubleConv(in_channels=1))

# 2. Load a subject's brain scan and atlas (both in 256³ 1mm FreeSurfer space)
brain = nib.load("NKI_FreeSurfer/freesurfer/<subj>/mri/brain.mgz").get_fdata().astype("float32")
atlas = nib.load("NKI_FreeSurfer/freesurfer/<subj>/mri/aparc+aseg.mgz").get_fdata().astype("int32")

# 3. Supply group-difference weights (e.g. t-statistics from group_diff.py output)
#    Shape: (n_features,) = (7680,) for CNN3D_DoubleConv
w = np.load("group_diff_t_stats.npy")

# 4. One call — returns a VoxelImportanceResult
result = compute_voxel_importance(
    model=model,
    brain_volume=brain,       # (H, W, D) numpy array — auto-batched internally
    importance_weights=w,
    atlas=atlas,
    lut="FS-LUT.txt",         # or pre-loaded dict from load_freesurfer_lut()
    device="cuda",            # falls back to CPU on OOM
    downsample_factor=2,      # 256³→128³; use 1 to skip
)

# 5. Inspect results
print(result.top_regions(n=10))          # top-10 regions by absolute importance
result.region_df.to_csv("importance.csv", index=False)

# 6. Save the voxel map as NIfTI for FSLeyes / ITK-SNAP
ref = nib.load("NKI_FreeSurfer/freesurfer/<subj>/mri/brain.mgz")
nib.save(nib.Nifti1Image(result.signed_map, ref.affine), "signed_importance.nii.gz")
nib.save(nib.Nifti1Image(result.abs_map,    ref.affine), "abs_importance.nii.gz")
```

### `VoxelImportanceResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `.signed_map` | `(H, W, D) float32` | Directional gradient map |
| `.abs_map` | `(H, W, D) float32` | Absolute sensitivity map |
| `.region_df` | `pd.DataFrame` | Per-region table, sorted by `abs_importance` |
| `.top_regions(n, by)` | method | Returns top-n rows of `region_df` |

### `region_df` columns

| Column | Description |
|--------|-------------|
| `label_id` | FreeSurfer aparc+aseg integer label |
| `region` | Human-readable region name from LUT |
| `n_voxels` | Number of voxels in this region |
| `signed_importance` | Mean signed gradient (direction-sensitive) |
| `abs_importance` | Mean absolute gradient (total sensitivity) |

### Low-level functions (for custom pipelines)

```python
from brainage_agg.analysis.voxel_attribution import (
    compute_attribution_map,   # backprop weights → (signed_map, abs_map)
    project_to_atlas,          # maps → {label_id: (mean, n_voxels)} dicts
    load_freesurfer_lut,       # parse FS-LUT.txt → {label_id: region_name}
)
```

See [brainage_agg/analysis/voxel_attribution.py](brainage_agg/analysis/voxel_attribution.py) for full docstrings.

A complete runnable example is at [examples/voxel_importance_example.py](examples/voxel_importance_example.py).

---

## Interactive Dashboard

Launch the voxel importance dashboard (requires extracted features and group difference results):

```bash
python -m brainage_agg.analysis.voxel_importance_dashboard
```

The dashboard shows:
- 3D brain surface with voxel importance overlay (nilearn)
- Scatter plot: CNN feature importance vs FS ROI effect size
- Bar chart: top-30 regions ranked by absolute importance

Dataset and aggregation method are selectable via dropdowns.

---

## Full Pipeline

### Preregistered voxel-attribution validation

The confirmatory campaign writes only under
`outputs/attribution_validation/<run_id>/`; it does not overwrite the
exploratory Phase 4b artifacts.

```bash
python -m brainage_agg.experiment.run_attribution_validation prepare \
  --run-id confirmatory_v1

# One array contains injection and architecture-prior cells and is capped at
# two simultaneous GPU jobs.
bash scripts/submit_attribution_validation.sh confirmatory_v1

# 199-permutation regional null, also capped at two simultaneous GPU jobs.
bash scripts/submit_attribution_null.sh confirmatory_v1

# Exits nonzero until every requested cell and artifact is hash-valid.
python -m brainage_agg.experiment.run_attribution_validation collect \
  --run-id confirmatory_v1

python -m brainage_agg.analysis.report_attribution_validation \
  --run-dir outputs/attribution_validation/confirmatory_v1 \
  --null-npz <nki-permutation-region-stats.npz> \
  --null-npz <ppmi-permutation-region-stats.npz> \
  --output outputs/attribution_validation/confirmatory_v1/paper
```

Each injected-effect cell attenuates the raw T1 volume before scaling, then
recomputes all `t1_rank_sobel` channels. Completed cells record resolved
configuration, commit state, subject IDs, RNG seeds, environment, hashes,
subject-level provenance, maps, and regional metrics. CUDA errors and missing
seeds, subjects, weights, maps, or permutations are fatal in confirmatory
mode. ROI concordance is secondary and is never used as a decision gate.

### 1. Extract CNN features

```bash
python scripts/extract_all_features.py --config configs/benchmark.yaml
```

Smoke run (4 subjects):

```bash
python scripts/extract_all_features.py --config configs/benchmark.yaml --limit 4
```

Multi-GPU with worker processes:

```bash
python scripts/extract_all_features.py \
  --config configs/benchmark.yaml \
  --gpus 0,1,2 \
  --seed-batch-size 3 \
  --num-workers 3 \
  --persistent-workers
```

Use `--gpus all` for every visible CUDA device.

### 2. Extract FreeSurfer ROI features

```bash
python scripts/extract_roi_features.py --config configs/benchmark.yaml
```

### 3. Run grouped evaluations

```bash
python scripts/run_all_evaluations.py --config configs/benchmark.yaml
```

### 4. Summarize and rank results

```bash
python scripts/summarize_results.py --results outputs/results/evaluation_results.csv
```

### 5. Group difference analysis (voxel importance weights)

Sequential (small experiments):

```bash
python -m brainage_agg.analysis.group_diff --dataset nki
python -m brainage_agg.analysis.group_diff --dataset ppmi
```

SLURM array (one job per extractor):

```bash
# Submit array — see brainage_agg/slurm/ for job scripts
sbatch brainage_agg/slurm/group_diff_array.sh

# After all jobs complete, merge partial results:
python -m brainage_agg.analysis.group_diff --dataset nki --merge
```

---

## Data Assumptions

The cohort CSV must contain at least:

```
id, participant_id, session, age, sex, timepoint, directory
```

Optional columns `days_since_enrollment`, `session_type`, `studies`, `study_num` are used when available.

FreeSurfer outputs are expected under:

```
NKI_FreeSurfer/freesurfer/<directory>/mri/brain.mgz
NKI_FreeSurfer/freesurfer/<directory>/mri/brainmask.mgz
NKI_FreeSurfer/freesurfer/<directory>/mri/aparc+aseg.mgz
```

Rows with missing image or mask files are dropped before feature extraction. Empty masks raise an error.

Edit [configs/benchmark.yaml](configs/benchmark.yaml) to set data paths, scalers, feature sets, CNN seeds, and evaluation parameters.

### Feature sets

```
t1                 — raw T1 intensity
t1_sobel           — T1 + Sobel edge magnitude
t1_rank_sobel      — rank-normalized T1 + Sobel
t1_median_sobel    — median-filtered T1 + Sobel
```

### Masked scalers

`zscore`, `robust`, `minmax` — all scaling statistics are computed inside the brain mask only; background voxels are set to zero.

### Outputs

```
outputs/features/features__scaler-<s>__channels-<c>__seed-<n>.npz   # feature arrays
outputs/features/features__scaler-<s>__channels-<c>__seed-<n>.json  # metadata + feature names
outputs/results/evaluation_results.csv
outputs/results/summary_by_feature_set.csv
```

---

## Leakage Control

All downstream evaluations use `GroupShuffleSplit` with subject identifiers as groups. The code asserts that no subject appears in both train and test splits before each model fit. Longitudinal pair evaluations are also split by subject. FS ROI Arm B/B2 results carry a `leakage_warning=True` flag because the Arm B target is derived from FS ROI features.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `torch`, `nibabel`, `nilearn`, `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `scipy`, `dash`, `plotly`, `tqdm`.
