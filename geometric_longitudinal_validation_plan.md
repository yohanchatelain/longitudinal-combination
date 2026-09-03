# Geometric Longitudinal Validation Plan

**Status:** Proposed revised confirmatory protocol  
**Date:** 2026-09-02  
**Objective:** Test whether a frozen, randomly initialized CNN measures known longitudinal anatomical change more accurately and robustly than an identically structured trained CNN, FastSurfer-based volumetric segmentation, and a classical deformation-based morphometry baseline.

**Amendment (2026-09-02):** the original design used FreeSurfer `recon-all` (cross-sectional and longitudinal stream) as the classical, non-learned comparator. Running `recon-all` at the scale this study needs was judged too time-consuming and is out of scope; FastSurfer (containerized, `deepmi/fastsurfer:cpu-latest` via Apptainer, `--seg_only` mode) replaces it as the volumetric-segmentation comparator throughout this document. This is not a drop-in substitution — see §1's revised motivation.

## 1. Motivation and scope

An untrained CNN cannot learn dataset-specific artifacts through weight optimization. It can nevertheless remain sensitive to interpolation, registration, preprocessing, floating-point differences, random initialization, and architectural priors. The proposed experiment therefore distinguishes two mechanisms:

1. intrinsic numerical sensitivity of an architecture and preprocessing pipeline; and
2. amplification of nuisance signals caused by supervised training.

**Consequence of the FastSurfer amendment:** the classical-vs-learned contrast this distinction depends on is no longer carried by "FreeSurfer vs. CNN." FastSurfer is itself a trained CNN-based segmentation tool — prior work from this lab (arXiv:2509.05238) found it exhibits training-time numerical variability comparable to, and in cortical regions exceeding, classical FreeSurfer. It belongs on the *learned* side of this study's distinction, alongside the trained CNN, not on the classical side. The only remaining classical (non-learned) comparator in this design is the deformation-based morphometry baseline (§5.4). Results should be interpreted accordingly: a FastSurfer-vs-frozen-CNN comparison tests mechanism (1) and (2) jointly across two different learned architectures, not learned-vs-classical.

The cancelled `confirmatory_v1` attribution array used raw-T1 intensity attenuation. Its 26 completed cells remain useful exploratory diagnostics, but intensity attenuation is not a known geometric longitudinal change and cannot establish geometric measurement accuracy. Those results must remain separate from the new confirmatory analysis.

The revised protocol is staged to provide an initial answer within approximately one week rather than running the original 1,004-cell campaign for several months.

## 2. Primary scientific question

Does a frozen randomly initialized CNN recover a known longitudinal geometric change more accurately and robustly than:

1. the same CNN architecture after supervised training;
2. FastSurfer-derived ROI volumes (a learned, DL-based comparator — see §1 amendment) analyzed with the same statistics originally specified for FreeSurfer; and
3. a conventional deformation-based morphometry baseline?

The primary comparison is paired within subject. Each real baseline image is used to create a synthetic follow-up with a precisely known deformation.

## 3. Exact geometric ground truth

### 3.1 Synthetic longitudinal conditions

| Condition | Known truth | Purpose |
|---|---:|---|
| Exact duplicate | 0% change everywhere | Basic false-positive test |
| Resampling sham | 0% anatomical change | Interpolation and registration sensitivity |
| Hippocampal contraction | 2%, 5%, and 10% | Localized, biologically plausible atrophy |
| Hippocampal expansion | +5% | Direction/sign recovery |
| Cerebellar-white-matter contraction | 5% | Adversarial tissue-boundary and architecture-prior control |
| Global scaling on a small subset | Exactly specified | End-to-end magnitude sanity check |

A smooth, invertible deformation field is applied only to the synthetic follow-up. Intensities move with the anatomy; intensity is not simply attenuated inside the target ROI.

### 3.2 Numerical definition of truth

For a target region \(R\) and deformation \(\phi\), the exact percentage volume change is calculated from the deformation Jacobian:

\[
\Delta V_{\mathrm{true}}
=
100\frac{\int_R \det(J_\phi(x))\,dx - V_R}{V_R}.
\]

The realized Jacobian integral, rather than only the requested nominal amplitude, is the ground truth used in analysis.

### 3.3 Avoiding circularity

The deformation mask must not be generated from the same segmentation tool being evaluated (FastSurfer, per §1's amendment). It should be defined by an independent atlas, a manually validated segmentation, or a fixed template-space mask. FastSurfer is credited only when new processing of the synthetic follow-up recovers the independently known change.

### 3.4 Deformation quality control

Every generated transformation must satisfy the following checks:

- positive Jacobian determinant everywhere;
- realized target-volume change within a predefined tolerance of the requested change;
- smooth transition at the target boundary;
- negligible change outside the target and transition band;
- identical image grid and metadata where required by the comparison;
- visual quality control for a random subset of subjects and every condition.

Failed transformations are regenerated or excluded according to a recorded, condition-blind rule.

## 4. Subjects and data separation

The fast first stage uses:

- 20 NKI validation subjects;
- 20 PPMI validation subjects;
- no overlap with subjects used to train the supervised CNNs;
- no overlap with pilot or calibration subjects.

A separate pilot set of approximately 20 subjects is used for transformation QC, runtime measurement, and calibration of any magnitude decoder. Pilot subjects are never included in confirmatory estimates.

If the first-stage confidence intervals are inconclusive, the validation sample expands to 50 subjects per cohort. This expansion is not launched automatically.

Subject selection, exclusions, cohort assignment, random seeds, and data hashes are recorded in a frozen run manifest before confirmatory analysis.

## 5. Methods

### 5.1 Frozen untrained CNN

- Randomly initialized convolutional weights remain frozen.
- Architectures: `double_conv` and `cov_pool`.
- Five independent initialization seeds.
- The same preprocessing, spatial resolution, and input channels as the trained CNNs.
- Longitudinal features are computed as paired follow-up-minus-baseline changes.

Random seeds quantify method variability; they are not treated as additional subjects.

### 5.2 Trained CNN

The trained comparator uses the same architectures and preprocessing. Only the learned weights differ.

- Use age-trained checkpoints and, if independently available, disease/status-trained checkpoints.
- Use multiple training seeds where practical.
- Do not train or fine-tune on the synthetic validation transformations.
- Ensure no subject overlap between model training and validation.

This isolates the effect of optimization from architecture and downstream analysis.

### 5.3 FastSurfer

Segmentation only (`--seg_only`, FastSurferVINN — equivalent to FreeSurfer's `aparc.DKTatlas+aseg.mgz`), run via the containerized `deepmi/fastsurfer:cpu-latest` image (Apptainer, no internet needed on compute nodes once pulled once to `containers/fastsurfer-cpu.sif`). No surface pipeline, no FreeSurfer license required — both are only needed for `recon-surf`, which this design does not use.

**Each time point is processed independently.** FreeSurfer's `-long` stream builds a subject-specific unbiased template from both time points jointly, specifically to suppress within-subject registration/segmentation noise (Reuter et al. 2012) — FastSurfer has no equivalent longitudinal mode. This design therefore has no comparator that benefits from that noise-suppression machinery; independent per-timepoint FastSurfer segmentation is expected to be noisier on the zero-change calibration experiment (§6.1) than a true `-long` stream would have been. Report this explicitly as a limitation of the amendment, not as a property of FastSurfer being tested — the study is no longer able to ask "does the field's best longitudinal-noise-control method beat a frozen CNN," only "does independent per-timepoint DL segmentation beat a frozen CNN."

Primary measurements are bilateral hippocampal and cerebellar-white-matter volumes, with intracranial-volume-adjusted results as a sensitivity analysis. Unaffected ROIs serve as spatial negative controls.

The primary analysis is a paired one-sample test of subject-level percentage change. A mixed-effects sensitivity analysis uses:

\[
y_{it} \sim \mathrm{time} \times \mathrm{condition} + (1|\mathrm{subject}).
\]

Paired \(t\)-tests and robust or Wilcoxon sensitivity results are reported. Statistical significance alone does not establish accuracy because the numerical truth is known.

### 5.4 Deformation-based morphometry

A conventional registration/Jacobian measurement is included as an image-based reference. It provides a useful upper benchmark under conditions favorable to registration.

## 6. Experiments

### 6.1 Experiment 1: zero-change calibration

Use exact duplicates and resampling shams to measure:

- mean apparent change;
- standard deviation of apparent change;
- 95th percentile absolute drift;
- false-positive rate at the preregistered threshold;
- consistency across cohorts, architectures, and random seeds.

A method that detects the 5% deformation but also reports change in sham pairs is not considered reliable.

Proposed calibration gate:

- false-positive rate at or below 5%;
- upper confidence bound at or below 10%;
- no consistent spatial localization inside the sham target mask.

### 6.2 Experiment 2: detection and dose response

Use 2%, 5%, and 10% hippocampal contractions to test:

- separation of true change from sham;
- monotonic response to increasing deformation;
- recovery of the correct change direction;
- sample size required to detect each dose.

Metrics:

- standardized separation from the sham distribution;
- sign accuracy;
- Spearman correlation with the realized Jacobian-derived dose;
- calibration slope;
- subject-level discrimination AUROC;
- power at sample sizes 10, 20, 30, and 40.

Sample-size curves are calculated by nested resampling of completed results, not by rerunning the image pipeline.

### 6.3 Experiment 3: magnitude accuracy

FastSurfer produces measurements in physical units, whereas frozen CNN features do not inherently represent percentage volume. Two CNN analyses are therefore reported separately.

The training-free primary analysis uses feature-change scores and tests detection, direction, and dose ranking.

A secondary analysis fits a regularized linear decoder from frozen feature changes to percentage volume change. The decoder is trained only on separate synthetic calibration subjects and evaluated on untouched validation subjects. This method is described as a **frozen random encoder with a calibrated linear readout**, not as a wholly untrained pipeline.

Metrics:

- mean absolute error;
- root mean squared error;
- signed bias;
- calibration slope and intercept;
- 95% interval coverage;
- error at each deformation amplitude.

### 6.4 Experiment 4: spatial localization

At the 5% amplitude, compare each attribution map with the known deformation support.

The primary CNN attribution uses t-statistic weighting only. SHAP and hybrid weighting are reserved for a small sensitivity subset after the primary result is available.

Metrics:

- voxel AUROC;
- normalized AUPRC lift over target prevalence;
- Dice overlap at fixed top-\(k\)% thresholds;
- center-of-mass localization error;
- fraction of attribution inside the true target;
- bilateral top-five ROI hit rate;
- attribution in unaffected contralateral and same-tissue ROIs.

Proposed localization gates retain the current attribution-validation thresholds:

- voxel AUROC at least 0.75;
- AUROC lower confidence bound above 0.50;
- normalized AUPRC lift at least 2.0;
- AUPRC-lift lower confidence bound at least 1.0;
- bilateral top-five ROI hit rate at least 0.80.

### 6.5 Experiment 5: architecture-prior control

Repeat the 5% deformation in cerebellar white matter. This target is an adversarial control because untrained convolutional architectures may emphasize tissue boundaries and edges.

The direction of the result must be consistent across:

- NKI and PPMI;
- `double_conv` and `cov_pool`;
- most random initialization seeds.

Report median performance, dispersion, and the worst-performing seed rather than selecting the best initialization.

### 6.6 Experiment 6: learned numerical-artifact stress test

This secondary experiment tests the proposed mechanism directly. Create two versions of an independent supervised training set:

1. a balanced set in which numerical or resampling variants are independent of the training label; and
2. a deliberately confounded set in which the numerical variant is correlated with the label.

Train otherwise identical CNNs and evaluate them on zero-anatomical-change longitudinal pairs where only the numerical-processing variant changes.

Compare:

- trained-balanced CNN;
- trained-confounded CNN;
- frozen random CNN;
- FastSurfer;
- deformation-based baseline.

If the confounded trained model produces substantially more false longitudinal signal, optimization can amplify numerical artifacts. If the frozen random CNN also reacts strongly, intrinsic architectural sensitivity is a major contributor. This deliberate stress test is not used to characterize ordinary trained models as inherently biased.

## 7. Reduced computational design

The redesigned primary CNN grid is:

\[
2\ \mathrm{cohorts}
\times 2\ \mathrm{architectures}
\times 2\ \mathrm{targets}
\times 5\ \mathrm{seeds}
= 40\ \mathrm{cells}.
\]

Each cell processes sham and all requested amplitudes from shared cached inputs. Amplitude and longitudinal aggregation are not separate SLURM jobs.

Computational reductions:

- annualized within-person change is the primary longitudinal endpoint;
- `mean` aggregation is removed from the primary grid;
- t-statistic attribution is primary;
- every scan is preprocessed once;
- generated deformations, channels, and frozen-CNN activations are cached;
- the same synthetic images are reused across methods;
- FastSurfer segmentation (containerized, `--seg_only`) runs as a CPU array — no longer the dominant runtime driver now that full `recon-all` is out of scope (§1 amendment): FastSurfer segmentation typically completes in minutes per scan, versus `recon-all`'s 8-14 hours;
- SHAP, hybrid attribution, additional ROIs, and larger samples are conditional follow-ups.

## 8. Statistical decision rules

A method is eligible for comparison only after passing zero-change calibration.

The untrained CNN is considered superior to a comparator only if:

1. the paired error difference favors the untrained CNN;
2. the subject-clustered 95% bootstrap interval excludes zero;
3. the improvement is practically meaningful, such as at least a 20% reduction in magnitude or localization error;
4. the false-positive rate is no worse;
5. the direction is consistent in NKI and PPMI; and
6. the result is not driven by one favorable random seed.

If differences are small, perform an equivalence analysis using a prespecified margin. Failure to reject a null hypothesis is not interpreted as equivalence.

Bootstrap resampling is clustered by subject. Cohort, architecture, target, amplitude, and CNN seed remain explicit sources of variation. Primary endpoints and decision gates are fixed before looking at confirmatory outcomes.

## 9. Staged execution and stopping rule

### Initial stage

1. **Day 1:** Implement and QC deformation generation; freeze subjects and manifests.
2. **Day 2:** Generate synthetic follow-ups and shared preprocessing caches.
3. **Days 2–3:** Run the 20-subject-per-cohort zero-change and 5% core experiment.
4. **Days 3–5:** Run dose response, attribution, and FastSurfer processing.
5. **Day 6:** Collect results, compute confidence intervals, and apply gates.
6. **Day 7:** Stop with a conclusion or approve only the specific expansion required.

The intended initial turnaround is approximately 3–7 days. FastSurfer throughput is no longer expected to be the binding constraint (§7); cluster availability for the CNN grid and deformation pipeline is the more likely driver now.

### Expansion rule

Expand from 20 to 50 subjects per cohort only when the first-stage confidence interval overlaps a prespecified superiority or equivalence boundary. Do not expand merely because a preferred method did not win.

## 10. Relationship to the cancelled campaign

SLURM array `363957` was cancelled after 26 cells completed. Those cells used intensity injection and are retained without alteration as exploratory results. They must be labeled as such and must not be pooled with the geometric confirmatory experiment.

The geometric experiment should use a new run identifier and a separately frozen configuration and manifest. The existing `confirmatory_v1` outputs remain an auditable record of the earlier design.

