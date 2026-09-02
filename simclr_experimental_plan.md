# SimCLR Trained-Comparator Experimental Plan

**Status:** Proposed  
**Date:** 2026-09-02  
**Objective:** Use the 3D-Neuro-SimCLR foundation model (Kaczmarek et al., arXiv:2509.10620, `github.com/emilykaczmarek/3D-Neuro-SimCLR`) as the trained-CNN comparator against the existing frozen un-CNN pipeline (`double_conv`/`cov_pool`), without building a supervised training pipeline from scratch, and use its trained/randomly-reinitialized weight pair as a fast substitute for the training-artifact mechanism test originally scoped as Experiment 6 in `geometric_longitudinal_validation_plan.md`.

This plan does not replace `geometric_longitudinal_validation_plan.md`; it specifies a narrower, faster-to-execute sub-project that can run in parallel, using an off-the-shelf checkpoint instead of the missing supervised-training infrastructure identified in that plan's review.

## 1. Scope decision: NKI only

This trained-vs-untrained comparator arm runs on **NKI validation subjects exclusively**. PPMI is excluded.

**Reason (hard constraint, not a TODO):** SimCLR's pretraining set (arXiv:2509.10620, Table 1) includes PPMI directly — 3,124 patients, 5,618 T1 scans, used in the 11-dataset training pool without holdout. No subject-ID exclusion list has been published to verify whether any of this project's PPMI validation subjects were among those 3,124. Using SimCLR on PPMI validation data therefore carries an unverifiable and plausibly nonzero train/test leakage risk. There is no cheap way to rule this out, so PPMI is excluded from this comparator arm rather than run with a caveat.

The PPMI arm of the trained-vs-untrained comparison stays open. It is not addressed by this plan and requires either a different trained comparator with a verifiably clean PPMI boundary, or a from-scratch trained checkpoint on data with a known subject list.

## 2. Prerequisite gate: random-init validity check

`SimCLR`'s backbone (`simclr/modules/resnet.py`) uses `nn.BatchNorm3d` throughout with default `track_running_stats=True`, confirmed by inspecting the cloned repository. A naive `pretrained=False`-style reinitialization leaves BatchNorm running statistics at PyTorch's defaults (running mean 0, running variance 1), which were never adapted to the activation distributions random convolution weights actually produce. Across an 18-layer ResNet-18 this is a known failure mode for random CNNs (saturating or degenerate deep features), which is precisely why un-CNN's own architecture (Encin et al., bioRxiv, `10.64898/2026.06.07.730652`) avoids depending on learned normalization statistics and compensates instead with input-level rank/Sobel channels.

This gate must pass before any trained-vs-untrained comparison (§5) proceeds. It is not optional and not deferrable.

### 2.1 Check design

Run on the pilot subject set (n = 20, NKI adult cohort, disjoint from confirmatory validation subjects — same pilot/confirmatory separation as `geometric_longitudinal_validation_plan.md` §4).

1. **Layer-wise activation statistics.** For every `BatchNorm3d` layer in the randomly-initialized backbone, run a forward pass over the pilot batch and record per-channel activation variance.
   - *Dead-channel rate:* fraction of channels with activation variance below 1e-6.
   - *Numerical validity:* presence of `NaN`/`Inf` anywhere in the pooled output features.
2. **Linear-probe sanity check.** Fit ridge regression from covariance-pooled random-backbone features to chronological age on the pilot set (leave-one-out or 5-fold, matching un-CNN's own evaluation protocol). This is not expected to approach the trained model's accuracy — it only needs to confirm the untrained features carry *some* anatomical signal, consistent with the general finding (Saxe et al., cited in the un-CNN paper) that untrained CNNs can extract non-trivial structure.

### 2.2 Gate criteria

| Check | Pass condition |
|---|---:|
| Dead-channel rate (all `BatchNorm3d` layers, pooled) | < 5% |
| `NaN`/`Inf` in pooled features | none, across all pilot subjects |
| Linear-probe age R² vs. permutation null | R² > 0, permutation p < 0.05 (1000 permutations) |

**If the gate fails:** attempt a documented fix — recompute BatchNorm running statistics via a forward pass in `train()` mode over the pilot set, then freeze and re-run the gate once. If it still fails, SimCLR-reinit is not usable as the untrained arm of this comparison; fall back to un-CNN as the sole untrained comparator and drop the matched-pair design in §5 in favor of the architecture-mismatched comparison (un-CNN vs. SimCLR-trained only, reported as such, not as a training ablation).

## 3. Preprocessing compatibility

SimCLR was pretrained using TurboPrep with rigid registration to the MNI152 ICBM 2009c nonlinear symmetric template. Exact invocation, per the model repository's README:

```
turboprep $T1_FILE $OUTPUT_DIR $T1_TEMPLATE -m t1 -r r
```

- `$T1_TEMPLATE` = `mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii`
- `-m t1`: T1-modality intensity normalization
- `-r r`: **rigid** registration (not affine, not nonlinear)

This must be reproduced exactly for any image fed to the SimCLR backbone — a different registration mode, template, or intensity-normalization step is an unverified distribution shift relative to what the model saw during pretraining.

**Same tool family as this repo's own preprocessing (both use TurboPrep), which is why the mismatch risk here is lower than it was for AnatCL's CAT12/VBM path — but not eliminated.** Two specific gaps remain:

1. This repo's existing un-CNN pipeline (per the bioRxiv paper's methods) uses ANTs-based registration to age-appropriate Fonov templates (separate pediatric and adult templates), not a single fixed adult MNI152 template. SimCLR uses one adult template for all ages. Applying SimCLR to NKI's child cohort (`Longitudinal_Child`, ages 6-20) means registering a child brain to an adult-only template with a pipeline never validated for that use.
   - **Decision: restrict this comparator arm to `Longitudinal_Adult` NKI subjects (ages 38-74) only.** Do not run SimCLR on the child cohort until the pediatric-template mismatch is separately investigated.
2. The exact per-volume intensity transform applied immediately before the tensor enters the network (beyond TurboPrep's own T1 normalization — e.g., any additional z-scoring or clipping in the SimCLR dataloader) is not confirmed from the README alone. Verify against `simclr/main_eval.py` / the dataset class before running anything (§7).

Build a dedicated preprocessing branch that produces SimCLR-compatible inputs (TurboPrep rigid-to-MNI152) separately from the existing un-CNN multi-channel (t1/rank/sobel, native-space) pipeline. These are two different preprocessing outputs from the same source scans, not a shared step.

## 4. Registration-free design integration

Per this project's broader goal of isolating numerical/training artifacts from registration artifacts, split experiments into two tracks.

### 4.1 Track A — primary endpoints (detection, dose-response, magnitude)

Native-space, no population/template registration:
- SynthStrip skull-strip (native space, no template)
- N4 bias-field correction (native space)
- Isotropic resampling to a fixed voxel size (grid resampling only, no anatomical warping)
- Header-based canonical reorientation (deterministic, uses the NIfTI affine already present — not an optimization-based registration)

Baseline and follow-up run **independently** through each backbone; the longitudinal signal is the difference of the pooled (covariance-pooled) feature vectors. No cross-timepoint alignment step exists in this track — covariance pooling gives approximate translation invariance, so no explicit correspondence between the two timepoints' voxel grids is required.

**Asymmetry that must be reported, not hidden:** un-CNN can run on this fully native-space input as designed. SimCLR cannot — it was pretrained exclusively on TurboPrep's rigid-to-MNI152 output, and feeding it raw native-space images pushes it far outside its training distribution, which would be a worse confound than the registration mismatch itself.

**Decision:** SimCLR uses its own trained preprocessing (TurboPrep rigid-to-MNI152) in Track A; un-CNN uses native-space input in Track A. This is a deliberate, documented asymmetry, not an oversight. Consequently:
- **un-CNN vs. SimCLR(trained)** is a bundled architecture + preprocessing + training comparison. Do not interpret any difference here as isolating a single mechanism.
- **SimCLR(trained) vs. SimCLR(untrained-via-reinit)** is the comparison that stays clean — both run on identical TurboPrep-registered input, so the only difference is whether the weights were optimized. This is the pair that carries the mechanism claim (§5), not the un-CNN comparison.

### 4.2 Track B — spatial localization only (matches Experiment 4 / §6.4 style)

Add a single **rigid, subject-specific** coregistration (follow-up → that subject's own baseline — not to a population template) on top of whichever per-model preprocessing branch is already in use (native for un-CNN, TurboPrep-MNI for SimCLR). This is the minimum alignment needed for voxel-level attribution comparison (Dice overlap, center-of-mass error) and is isolated from Track A so any localization-quality difference cannot be attributed to population-registration noise.

## 5. Trained-vs-untrained matched-pair mechanism experiment

Replaces `geometric_longitudinal_validation_plan.md` §6.6's "train two CNNs from scratch under balanced/confounded label conditions" design, which requires supervised-training infrastructure this repo does not have. This is a narrower, adapted version — the scope change is explicit below, not implicit.

**Arms:** `SimCLR(pretrained=True)` ("trained") vs. `SimCLR(pretrained=False)`, gated by §2 ("untrained-via-reinit"). Same architecture, same weight topology, only difference is whether the weights were optimized.

**What changed from the original §6.6 design:** the original mechanism test perturbed *training* (different seeds/confound conditions during optimization, per arXiv:2509.05238) and asked whether training amplifies numerical artifacts. Without a training pipeline, this plan instead perturbs *inference*: it asks whether the trained model's fixed, optimized weights are more sensitive to inference-time numerical noise than random weights are. These are related but distinct claims, and results must be reported as testing the inference-sensitivity question, not the training-stochasticity question the arXiv paper answered.

**Numerical-variant manipulation:** reuse Fuzzy PyTorch (`github.com/big-data-lab-team/fuzzy-pytorch`), the Monte Carlo Arithmetic (MCA) floating-point perturbation tool already used for FastSurfer in the companion paper from the same lab (arXiv:2509.05238). Applied at inference, not training.

### 5.1 Zero-change pairs

Reuse the exact-duplicate and resampling-sham constructions from `geometric_longitudinal_validation_plan.md` §3.1 — both are known-zero-anatomical-change conditions, so any apparent longitudinal signal is by definition spurious.

### 5.2 Procedure

For each pilot subject (n = 20, `Longitudinal_Adult`, disjoint from confirmatory subjects):
1. Construct a zero-change pair (duplicate + resampling sham).
2. Run both arms (trained, untrained-via-reinit) at default IEEE precision — baseline apparent-change value.
3. Run both arms across K = 10 MCA-perturbed repetitions (matching the repetition count used in arXiv:2509.05238), recording apparent-change magnitude (pooled-feature difference) per repetition.
4. Compute per-subject, per-arm dispersion (standard deviation) of apparent change across the 10 repetitions.

### 5.3 Metrics

- Standard deviation of apparent change under MCA perturbation, per arm.
- Ratio `SD_trained / SD_untrained`, with subject-clustered bootstrap 95% CI (≥ 2000 resamples, matching `bootstrap_samples` in `configs/attribution_validation.yaml`).
- Fraction of MCA repetitions exceeding a fixed apparent-change threshold (a false-positive-like rate), per arm.

## 6. Decision gates and timeline

### 6.1 Gates

| Gate | Criterion | Pass condition |
|---|---|---:|
| 0 — random-init validity (§2) | Dead-channel rate, NaN/Inf, linear-probe R² | All three conditions in §2.2 |
| 1 — zero-change calibration | Mean apparent change ≈ 0; FPR at preregistered threshold; upper CI bound | FPR ≤ 5%, UCB ≤ 10% (per arm, matching `geometric_longitudinal_validation_plan.md` §6.1) |
| 2 — mechanism decision | `SD_trained / SD_untrained` bootstrap 95% CI | Lower bound > 1.0 required to conclude training amplifies inference-time numerical sensitivity; CI spanning 1.0 is a null result, not evidence either way |
| 3 — reproducibility | Direction of Gate 2 result | Consistent between pilot (n=20) and expanded confirmatory sample before any claim is finalized |

Gate 0 blocks everything downstream. Gates 1-2 are computed independently; a Gate 1 failure (either arm shows non-trivial false-positive drift on zero-change pairs) invalidates that arm as a reliable measurement instrument regardless of Gate 2's outcome, per the same logic as `geometric_longitudinal_validation_plan.md` §8's eligibility rule.

### 6.2 Timeline

| Day | Task |
|---|---|
| 1 | Build SimCLR-compatible TurboPrep preprocessing branch for NKI `Longitudinal_Adult` pilot subjects (n=20). Implement Gate 0 check code. |
| 2 | Run Gate 0. If pass, proceed. If fail, apply the running-statistics recomputation fix (§2.2) and re-check — do not proceed past this day without a pass or a documented fallback to un-CNN-only comparison. |
| 3 | Wire Fuzzy PyTorch MCA perturbation into SimCLR inference. Build zero-change pair construction (duplicate + resampling sham) for the pilot set. |
| 4 | Run the mechanism experiment (§5) on the pilot set; compute Gates 1-2. |
| 5 | If pilot results are directionally clear and gates resolve decisively, expand to the full NKI `Longitudinal_Adult` confirmatory validation sample; otherwise stop with a documented null/inconclusive result rather than expanding on ambiguous grounds. |

Total: ~5 days if Gate 0 passes on the first attempt; +1 day if the running-statistics fix is required.

## 7. Open risks and unknowns

These must be resolved before execution, not discovered during it.

1. **Possible indirect PPMI *or NKI* leakage via CoRR.** SimCLR's Table 1 includes CoRR (Consortium for Reliability and Reproducibility, Zuo et al. 2014) at 1,416 patients. CoRR is a multi-site aggregation project that has historically included the NKI-Rockland Sample as one of its constituent sites. If any of this repo's NKI validation subjects are present in CoRR's release, the "NKI is clean" conclusion reached earlier in this project's review is not verified — it is asserted from the *named* dataset list only. **This must be checked against CoRR's actual site/subject manifest before Gate 0 is treated as sufficient scientific grounding for §1's NKI-only scope; if overlap exists, the same restriction problem this plan solved for PPMI reappears for NKI.**
2. **Exact inference-time normalization transform.** TurboPrep's `-m t1` output normalization is confirmed from the README; any additional transform applied in `simclr/main_eval.py` or the dataset class before the tensor reaches the network is not yet confirmed. Read the actual dataloader code, not just documentation, before running inference.
3. **Checkpoint format / PyTorch version compatibility.** The release checkpoint is a `.tar` archive; whether it unpacks to a plain `state_dict` loadable under this repo's installed PyTorch version is unverified. `requirements.txt` pins `torch` with no version constraint, so compatibility is not currently reproducible across environments — pin a specific version before this becomes a dependency other code relies on.
4. **License compliance for any derived checkpoint.** MIT permits reuse and modification. If Gate 0's running-statistics-recomputation fix (§2.2) is applied and the resulting weights are saved or shared, retain the original license/attribution — this is expected to be a non-issue under MIT but has not been formally checked against the exact license text in the cloned repo.
5. **Fuzzy PyTorch operator coverage.** Built and validated for FastSurfer (2D slice-based U-Net) and an MNIST sanity check; not yet confirmed to cover every operation in SimCLR's 3D ResNet-18 (e.g., any custom or fused ops) without modification.
6. **Scope-narrowing from the original mechanism claim (§5).** The inference-time MCA-perturbation design tests a different (related, not identical) question from arXiv:2509.05238's training-time-variability result. Any write-up must state this precisely rather than presenting it as a direct replication of that paper's finding on a new architecture.
