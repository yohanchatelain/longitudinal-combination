# Registration-Free Fixed Features

Self-contained implementation of the architecture described in the repository
README. It deliberately does not reuse the existing FreeSurfer dataset class:
primary extraction starts from raw T1 images and explicit masks.

## Included

- canonical, interpolation-free orientation handling;
- affine, spacing, truncation, connected-component, and mask-volume QC;
- optional pinned SimpleITK N4 correction and robust masked normalization;
- physical-space solid-harmonic energy filters with orders 0–2;
- automatic CUDA acceleration for all first- and second-order convolutions;
- whole-brain and intrinsic radial-shell summaries with immutable names;
- non-redundant two-visit and multi-visit representations;
- reusable outer folds and paired nested elastic-net evaluation;
- immutable NPZ/JSON artifacts with hashes, environment, configuration, and QC;
- deterministic perturbation primitives and unit/vertical-slice tests.

The external brain-extraction command is intentionally outside this package.
The manifest must name an explicit mask for every raw scan; there is no
`image > 0` fallback.

## Manifest

```csv
subject_id,visit_id,elapsed_years,image_path,mask_path
sub-001,ses-01,0.0,/data/sub-001_ses-01_T1w.nii.gz,/data/sub-001_ses-01_mask.nii.gz
sub-001,ses-02,1.2,/data/sub-001_ses-02_T1w.nii.gz,/data/sub-001_ses-02_mask.nii.gz
```

Paths should be absolute. Each image and mask must share a grid and affine.

## Commands

Create the isolated CUDA 12.1 environment used on this cluster from the
repository root:

```bash
bash registration_free_fixed_features/scripts/create_gpu_env.sh
```

CUDA 12.1 is deliberate: the GPU nodes expose a CUDA 12.3-capable driver, while
the repository's existing environment currently contains a CUDA 13.0 PyTorch
build that cannot initialize on those nodes.

```bash
python -m registration_free_fixed_features.cli extract \
  --config registration_free_fixed_features/config.example.yaml \
  --manifest raw_manifest.csv \
  --output outputs/registration_free/rfff-1/visit_features.npz

python -m registration_free_fixed_features.cli make-folds \
  --config registration_free_fixed_features/config.example.yaml \
  --targets targets.csv --target-column age --task regression \
  --output outputs/registration_free/rfff-1/folds.csv

python -m registration_free_fixed_features.cli evaluate \
  --config registration_free_fixed_features/config.example.yaml \
  --features outputs/registration_free/rfff-1/visit_features.npz \
  --targets targets.csv --target-column age --task regression \
  --folds outputs/registration_free/rfff-1/folds.csv \
  --output-dir outputs/registration_free/rfff-1/evaluation
```

The example configuration uses `bias_backend: none` for local development.
Confirmatory mode rejects that setting and requires `simpleitk_n4`; install
SimpleITK before confirmatory extraction.

### GPU execution

`scattering.backend: auto` and `scattering.device: auto` select PyTorch/CUDA
when a visible GPU is available, otherwise they use the verified SciPy CPU
reference. To require a particular GPU rather than permit fallback, set:

```yaml
scattering:
  backend: torch
  device: cuda:0
  deterministic: true
```

The CUDA path batches every real/imaginary spherical-harmonic kernel for one
angular order. It uses direct `conv3d` for compact kernels and cached batched FFT
convolution for larger physical kernels. First-order tensors remain on the GPU
for second-order scattering; only completed orientation-energy fields are moved
to the CPU for mask-restricted summaries. TF32 and cuDNN benchmarking are
disabled in deterministic mode. The resolved backend, CUDA and PyTorch
versions, and GPU name are written into the feature artifact metadata.

Run the CUDA/SciPy agreement and timing smoke test on this cluster with:

```bash
sbatch registration_free_fixed_features/slurm/gpu_smoke.sbatch
```

The job writes both a SLURM log and a machine-readable JSON report under
`registration_free_fixed_features/slurm/logs/`.

The checked GPU validation on a Tesla T4 is saved at
`validation/gpu_smoke_t4.json`: the batched FFT backend produced 377 finite
features in 1.24 seconds versus 5.16 seconds for SciPy (4.15× faster), with
relative L2 error `1.31e-7` and 324 MiB peak allocated VRAM.

### Development pilot with current PPMI data

The current PPMI cohort contains raw T1 images but no native masks. For resource
testing only, build a small stratified manifest by resampling each corresponding
FreeSurfer brain mask onto the raw scanner grid using header geometry:

```bash
python -m registration_free_fixed_features.development_manifest \
  --cohort-csv PPMI_data/cohort_longitudinal.csv \
  --freesurfer-root ppmi \
  --subjects-per-group 2 \
  --output-dir registration_free_fixed_features/pilot/ppmi

sbatch registration_free_fixed_features/slurm/pilot_extract.sbatch
```

The manifest and every mask provenance record carry
`confirmatory_eligible: false`. These masks can establish runtime, memory, QC,
and schema behavior, but cannot support the registration-free scientific claim.

Generate scanner-grid masks independently from the raw images with the pinned
SynthStrip 1.8 container:

```bash
sbatch registration_free_fixed_features/slurm/pilot_synthstrip.sbatch
```

The SLURM entry point refuses to run if the 382,615,552-byte SIF differs from
SHA-256 `fea95be33f3e2d4102d513349b2d95266c8528015e6b4d708d2abbcf91e1e462`.
It writes a new manifest, full command/input/output provenance, mask QC, and
Dice/Jaccard comparisons against the development masks. The frozen mask method
is eligible for confirmatory use, but this eight-scan resource pilot remains
ineligible because its subjects were selected through FreeSurfer-output
availability. A confirmatory manifest must be selected from raw-data criteria
alone. The official 1.8 container contains a CPU-only PyTorch build, so mask
generation uses four CPU threads; this does not change the GPU scattering stage.

After mask QC succeeds, extract the matched native-mask feature artifact with:

```bash
sbatch registration_free_fixed_features/slurm/pilot_extract_synthstrip.sbatch
```

The checked eight-scan result is stored in
`validation/synthstrip_pilot_t4.json`. SynthStrip masks passed every QC rule and
had mean Dice 0.9630 against the development masks. The matched 377-feature
artifact was finite and schema-identical; mask substitution changed it by 1.50%
relative L2 overall (0.02–2.82% per scan), with correlation 0.99990. Full-volume
extraction took 82 seconds on a Tesla T4 and peaked at 7.62 GiB allocated VRAM.
The reusable `registration_free_fixed_features.compare_artifacts` command
produces the same schema/order checks and stability metrics for subsequent
preprocessing or perturbation variants.

Exercise the pinned SimpleITK 2.5.6 N4 parameters on the same scans before
freezing confirmatory preprocessing:

```bash
sbatch registration_free_fixed_features/slurm/pilot_extract_synthstrip_n4.sbatch
```

## Deliberate limits

- This implementation requires externally generated masks and audits them; it
  does not choose or train a brain extractor.
- Unsupported affine shear and spacing are rejected instead of resampled.
- Perturbation orchestration and numeric acceptance margins remain campaign
  configuration, not hidden defaults in the extractor.
- The solid-harmonic implementation favors a transparent physical-space
  reference over GPU throughput. Resource pilots should precede cohort-scale
  extraction.
