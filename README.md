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
