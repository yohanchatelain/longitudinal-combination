# Voxel Attribution Maps — LaTeX Subsection

Append after `\subsection{Weight initialization}`.

```latex
\subsection{Voxel Attribution Maps}

To identify which brain regions contribute most to the CNN feature embedding,
we project feature-level importance weights back to voxel space via gradient
backpropagation~\cite{simonyan2013deep}.
Given the frozen network mapping an input volume
$\mathbf{x} \in \mathbb{R}^{C \times H \times W \times D}$ to a
$d$-dimensional feature vector $\mathbf{f}(\mathbf{x})$, a vector of
importance weights $\mathbf{w} \in \mathbb{R}^d$ is first derived by
associating each feature dimension with a scalar quantity of interest (e.g.\
a per-feature $t$-statistic or regression coefficient).

A scalar summary of the feature space is then formed as
\begin{equation}
  S(\mathbf{x}) = \sum_{i=1}^{d} w_i \cdot f_i(\mathbf{x}),
\end{equation}
and the voxel attribution map is obtained as its gradient with respect to the
input:
\begin{equation}
  \mathbf{G}(\mathbf{x}) = \frac{\partial S}{\partial \mathbf{x}}
                          = \sum_{i=1}^{d} w_i \frac{\partial f_i}{\partial \mathbf{x}}.
\end{equation}
Because all network weights are fixed, this gradient reflects how local
intensity patterns modulate the weighted feature summary through the frozen
random basis of the untrained network.

Two complementary maps are derived from $\mathbf{G}$: the \emph{signed map}
$\bar{G} = \mathrm{mean}_c\,G_c$ retains sign, indicating whether a voxel
increases or decreases the scalar $S$; the \emph{absolute map}
$|\bar{G}| = \mathrm{mean}_c\,|G_c|$ captures total voxel sensitivity
regardless of direction.
Both maps are projected to brain regions by averaging voxel values within each
parcel of the FreeSurfer Desikan--Killiany \texttt{aparc+aseg} atlas, yielding
a per-region importance ranking that can be compared against volumetric effect
sizes derived from classical region-of-interest analysis.
```

## Results (draft — for review, not yet polished for submission)

Computed from the completed Phase 4b full sweep (`brainage_agg/outputs/` /
`brainage_agg/ppmi_outputs/` under `group_diff/voxel_importance/`; NKI
Child-vs-Adult, PPMI PD-vs-HC; 10 CNN seeds; group_perm null correction,
10 permutations). Region-level agreement between the CNN attribution map
(mean |attribution|) and the classical FreeSurfer ROI effect size
(max |Cohen's d| per region) across all 6 aggregations × 3 weight-derivation
methods × 2 datasets:

| Dataset | Aggregation | Weight | n | Pearson r | Spearman rho | Top region (mean abs attr) |
|---|---|---|---|---|---|---|
| NKI | mean | tstat | 105 | 0.175 | 0.141 | ctx-rh-rostralmiddlefrontal |
| NKI | mean | shap | 105 | 0.210 | 0.244 | Right-Cerebellum-White-Matter |
| NKI | mean | hybrid | 105 | 0.072 | 0.044 | ctx-rh-parsopercularis |
| NKI | concatenation | tstat | 106 | 0.062 | 0.086 | Right-Cerebellum-Cortex |
| NKI | concatenation | shap | 106 | 0.041 | -0.098 | ctx-lh-supramarginal |
| NKI | concatenation | hybrid | 106 | 0.145 | 0.106 | ctx-lh-inferiorparietal |
| NKI | annualized_rate | tstat | 106 | -0.103 | -0.073 | Right-Cerebellum-Cortex |
| NKI | annualized_rate | shap | 106 | -0.111 | -0.061 | Right-Cerebellum-Cortex |
| NKI | annualized_rate | hybrid | 106 | -0.109 | -0.027 | Right-Cerebellum-Cortex |
| NKI | lme_slope | tstat | 106 | 0.022 | -0.038 | Right-Cerebellum-White-Matter |
| NKI | lme_slope | shap | 106 | 0.099 | 0.106 | Right-Cerebellum-Cortex |
| NKI | lme_slope | hybrid | 106 | n/a (constant) | n/a (constant) | Left-Lateral-Ventricle |
| NKI | difference | tstat | 106 | -0.107 | -0.050 | Right-Cerebellum-Cortex |
| NKI | difference | shap | 106 | 0.177 | 0.216 | ctx-lh-frontalpole |
| NKI | difference | hybrid | 106 | -0.072 | -0.050 | Right-Cerebellum-Cortex |
| NKI | lme_slope_change | tstat | 106 | 0.030 | -0.041 | Right-Cerebellum-White-Matter |
| NKI | lme_slope_change | shap | 106 | 0.070 | 0.016 | ctx-lh-frontalpole |
| NKI | lme_slope_change | hybrid | 106 | n/a (constant) | n/a (constant) | Left-Lateral-Ventricle |
| PPMI | mean | tstat | 109 | -0.124 | -0.212 | ctx-lh-supramarginal |
| PPMI | mean | shap | 109 | -0.598 | -0.645 | CC_Central |
| PPMI | mean | hybrid | 109 | 0.020 | 0.070 | ctx-rh-inferiortemporal |
| PPMI | concatenation | tstat | 109 | -0.095 | -0.141 | ctx-lh-supramarginal |
| PPMI | concatenation | shap | 109 | -0.589 | -0.614 | CC_Anterior |
| PPMI | concatenation | hybrid | 109 | -0.189 | -0.222 | ctx-rh-entorhinal |
| PPMI | annualized_rate | tstat | 109 | 0.043 | 0.118 | Right-Cerebellum-Cortex |
| PPMI | annualized_rate | shap | 109 | -0.449 | -0.441 | CC_Mid_Posterior |
| PPMI | annualized_rate | hybrid | 109 | 0.003 | 0.041 | Left-Cerebellum-Cortex |
| PPMI | lme_slope | tstat | 109 | 0.354 | 0.367 | ctx-lh-frontalpole |
| PPMI | lme_slope | shap | 109 | -0.464 | -0.476 | CC_Posterior |
| PPMI | lme_slope | hybrid | 109 | 0.167 | 0.210 | Right-Cerebellum-White-Matter |
| PPMI | difference | tstat | 109 | -0.027 | 0.054 | Left-Cerebellum-Cortex |
| PPMI | difference | shap | 109 | -0.453 | -0.455 | CC_Mid_Posterior |
| PPMI | difference | hybrid | 109 | 0.065 | 0.070 | Left-Cerebellum-Cortex |
| PPMI | lme_slope_change | tstat | 109 | 0.354 | 0.367 | ctx-lh-frontalpole |
| PPMI | lme_slope_change | shap | 109 | -0.464 | -0.476 | CC_Posterior |
| PPMI | lme_slope_change | hybrid | 109 | 0.167 | 0.210 | Right-Cerebellum-White-Matter |

**Headline finding: agreement between the CNN attribution maps and classical
ROI effect sizes is weak overall**, and inconsistent in sign. Mean Pearson r
across the 34 non-degenerate cells is +0.05 (tstat), +0.03 (hybrid), and
**-0.21 (shap)** — i.e. the SHAP-derived weights are, on average, *anti*-correlated
with where the classical ROI analysis finds effects, most strongly so on
PPMI (r down to -0.60 for `mean`/shap). This does not by itself show the CNN
attribution is wrong — a frozen, untrained CNN has no reason to rediscover
the same effect-size ranking as a linear ROI model — but it means the
"CNN vs ROI agreement" comparison **cannot currently be reported as a
positive validation of the attribution method** without further caveats.

Three things to resolve before these numbers go in a paper draft:

1. **Cerebellum / corpus-callosum dominance.** 22 of the 36 (61%) top-attributed
   regions above are cerebellar or corpus-callosum labels — both are boundary
   regions prone to partial-volume and registration artifacts, and sit near
   the edge of the intracranial mask. Their disproportionate share of "top
   region by mean |attribution|" is more consistent with an architecture/edge
   gradient bias (of the kind the permutation-null correction is meant to
   remove) than with a genuine group-difference signal. Worth checking
   whether the null correction is fully suppressing this before trusting the
   region ranking.
2. **`w_hybrid` is degenerate across most of the sweep, not just two cells.**
   The two NKI cells above are constant (Pearson/Spearman undefined) because
   *every* seed's permutation importance collapsed to all-zero — confirmed
   from the sweep logs (`phase4b_284815_{3,5}.out`: `imp_nonzero_frac=0.000`,
   `train_acc=1.000`, 10/10 seeds). But pulling `imp_nonzero_frac` across all
   12 array-task logs shows this isn't isolated: **83 of 120 hybrid RF fits
   sweep-wide (69%) hit `imp_nonzero_frac=0.000`**, ranging from 2/10 seeds
   (task 2) to 10/10 (tasks 3, 5). `min_samples_leaf=5` — added specifically
   to prevent this (see `importance_weights.fit_rf` docstring) — is not
   sufficient at `d_agg` up to 23,040 with n≈50-250 subjects. This means
   `w_hybrid` results should be treated as unreliable **throughout** the
   sweep, not only in the two constant-map cells, and the hybrid row of the
   table above should not be reported without either (a) a stronger RF
   regularization pass and re-run, or (b) dropping `w_hybrid` from the
   comparison entirely.
3. **`lme_slope` and `lme_slope_change` rows are numerically identical**
   (both datasets, all three weight types) — this is expected, not a bug:
   `aggregations.lme_slope_change` is defined (`brainage_agg/agg/aggregations.py:112-123`)
   to return the *same* per-subject BLUP slope as `lme_slope`; Arm A vs Arm B
   differ only in the downstream age-prediction *target* (absolute age vs.
   Δt), not in the feature descriptor. Voxel attribution is derived from the
   feature matrix and group labels alone, so identical features →
   identical attribution maps. Worth a one-line footnote in the writeup so a
   reviewer doesn't flag it as a duplication bug the way it first looked here.

## BibTeX entry to add

```bibtex
@article{simonyan2013deep,
  title   = {Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps},
  author  = {Simonyan, Karen and Vedaldi, Andrea and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:1312.6034},
  year    = {2013},
}
```
