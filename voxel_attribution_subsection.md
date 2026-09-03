# Voxel Attribution: Confirmatory Methods, Results, and Limitations

## Methods

The six-aggregation, ten-seed voxel-attribution sweep was designated
exploratory. Confirmatory validation uses controlled spatial injections as
the primary ground truth. NKI and PPMI subjects are assigned deterministically
to disjoint pilot and validation splits, with balanced synthetic injected and
control groups. Smooth multiplicative attenuation is applied bilaterally to
putamen, hippocampus, superior-frontal cortex, or cerebellar white matter in
the raw T1 image, using a two-voxel internal taper. Scaling and all T1,
local-rank, and Sobel channels are recomputed after injection.

Candidate attenuation amplitudes are 2%, 5%, and 10%. The primary amplitude
is selected without inspecting attribution maps, using only the pilot split,
as the candidate nearest regional Cohen's d=0.8. Localization is then
evaluated on the held-out validation split for `double_conv` and `cov_pool`,
the `mean` and `annualized_rate` aggregations, ten fixed CNN seeds, and t-stat,
SHAP, and hybrid feature weights. Primary endpoints are voxel AUROC,
prevalence-normalized AUPRC, attribution-mass fraction inside the injected
mask, bilateral target-region rank, and top-five hit rate.

The cohort null uses 199 label permutations for five fixed `double_conv`
seeds and retains the aggregated regional statistic from every permutation.
One-sided empirical p-values use the Monte Carlo +1 correction; adjusted
p-values compare each observed statistic with the permutation maximum across
regions. Leave-one-permutation-out pseudo-null evaluations estimate the maxT
family-wise false-positive rate. Unit feature weights on the mean brain, a
constant masked input, and a phase-scrambled input quantify architecture and
input priors. Boundary bias is analyzed with each region's boundary-voxel
fraction, log volume, and tissue class rather than by excluding selected
named regions.

FreeSurfer ROI concordance is secondary because ROI effects and voxel
attributions describe different models. Raw, null-excess, and
empirical-standardized regional scores are reported with seed-bootstrap
intervals and permutations stratified jointly by hemisphere, tissue class,
and region-volume quintile. ROI concordance is not a decision gate.

## Exploratory results

Null correction reduced—but did not eliminate—the dominant edge prior:
cerebellar or corpus-callosum regions were top-ranked in 23/36 raw cells and
11/36 corrected cells. Across the corrected exploratory sweep, mean Pearson
ROI correlations were negative for t-stat (-0.269), SHAP (-0.127), and hybrid
(-0.098) weights. These results do not validate biological localization.

The previously incomplete NKI `lme_slope/hybrid` cell has now completed for
50 subjects and ten seeds. Among 106 matched regions, Pearson correlation
with FreeSurfer ROI effect magnitude was 0.262 for the raw absolute score,
0.268 for null-excess absolute score, and 0.450 for the empirical-standardized
absolute score. The raw top region was Right-Pallidum. This single completed
exploratory cell does not change the confirmatory decision policy and is not
interpreted as biological validation.

## Confirmatory decision policy

Localization passes only if, at the calibrated amplitude, voxel AUROC is at
least 0.75 with bootstrap lower bound above 0.50, normalized AUPRC lift is at
least 2 with lower bound above 1, and bilateral top-five hit rate is at least
80%. Null calibration additionally requires a maxT family-wise false-positive
rate no greater than 5% with 95% upper bound no greater than 10%. Target
specificity requires within-target map similarity to exceed both
between-target similarity and similarity to the unit-weight prior, with
paired-bootstrap lower bounds above zero. General claims require all three
gates in both cohorts and both architectures.

Until these gates pass, voxel attribution is described only as an
architecture/preprocessing diagnostic and highlighted regions are not
interpreted biologically.

## Limitations

Synthetic attenuation is a localization ground truth, not a complete model of
neurodegeneration or development. The cerebellar white-matter target is an
adversarial test of a known edge-sensitive prior and must not be read as
evidence for cerebellar biology. Atlas boundaries and tissue contrast can
still influence gradients after null correction. Finally, successful
technical localization would establish sensitivity to controlled effects,
not biological validity; biological concordance remains a separate empirical
question.
