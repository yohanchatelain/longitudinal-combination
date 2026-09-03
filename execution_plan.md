# Consolidated Execution Plan

**Status:** Orchestration layer over three existing documents — does not restate their content, only sequences it.
**Date:** 2026-09-02
**Inputs:**
- `geometric_longitudinal_validation_plan.md` — the science plan (Experiments 1-6, decision rules §8).
- `geometric_longitudinal_validation_architecture.md` — the implementation architecture (raw-vs-invariant CNN design, package layout, build order §7).
- `simclr_experimental_plan.md` — the fast substitute for the science plan's §6.6 mechanism test, using an off-the-shelf checkpoint instead of a from-scratch trained CNN.

## 1. Why these three need one sequencing layer

Each document was scoped independently and each says, correctly, what it does not cover:
- The science plan assumes a trained CNN and a working deformation/FreeSurfer pipeline exist (§5.2-§5.4) — they don't (confirmed independently in this review and in the architecture doc §0).
- The architecture doc adds a whole new engineered mechanism (raw-vs-invariant, §2) on top of the science plan without touching the trained-CNN gap.
- The SimCLR plan closes part of the trained-CNN gap, but only for NKI `Longitudinal_Adult`, only for the §6.6-style mechanism question, and explicitly does not replace the science plan's Experiments 1-5 comparator.

None of the three, read alone, tells you what to build first or which parts can run in parallel. That's this document's only job.

## 2. Dependency graph

```
                    ┌─ 0a. SimCLR Gate 0 + shared zero-change pairs ──┐
                    │      (simclr_plan §2, §5.1)                    │
                    │                                                 ▼
  Day 1 start ──────┤                                    Phase 1: SimCLR mechanism
                    │                                    experiment (simclr_plan §5)
                    ├─ 0b. geometric/config.py +                      │
                    │      determinism.py (gate 0)                    │   (independent branch —
                    │      (architecture §7 steps 1-2)                │    no dependency on 0c/0d/
                    │                                                  │    Phase 2/Phase 3)
                    ├─ 0c. geometric/deformation.py +
                    │      qc.py + synthesis.py
                    │      (architecture §7 step 3, pure NumPy/SciPy)
                    │
                    └─ 0d. Blocking decisions (not builds):
                           - FreeSurfer availability (cluster-ops ticket)
                           - Registration library for DBM baseline
                           - Invariance-layer filter family
                           - CNN-training label scope

  0b + 0c done ──────────► Phase 2: invariance_layer.py + self_calibration.py,
                            calibrated on pilot split, raw-vs-invariant gate
                            run on Experiment 1 ALONE first (architecture §2.2)
                                    │
  Phase 2 gate passes ─────────────┤
  0d resolved ──────────────────────► Phase 3: full Experiments 1-6 grid
                                       (science plan §6-§9)
                                       Trained-CNN arm: OPEN — see §5 below
```

Phase 1 is the only branch with no dependency on the deformation module, FreeSurfer, or registration tooling. It is the fastest path to a first empirical result on the core hypothesis and should start immediately, in parallel with 0b/0c/0d, not after them.

## 3. Phase definitions

### Phase 0 — Foundations (parallel tracks, start Day 1)

| Track | Owner scope | Depends on | Blocks |
|---|---|---|---|
| 0a | SimCLR Gate 0 (random-init validity, simclr_plan §2) + zero-change pair construction (duplicate + resampling sham, science plan §3.1 — build once, reused by both Phase 1 and Phase 3) | nothing | Phase 1 |
| 0b | `geometric/config.py`, `determinism.py` reproducibility self-test gate (architecture §2.1, §7 steps 1-2) | nothing | Phase 2 |
| 0c | `geometric/deformation.py`, `qc.py`, `synthesis.py` — Jacobian-derived ground truth (architecture §7 step 3) | nothing | Phase 2, Phase 3 |
| 0d | Decisions, not code: file the FreeSurfer cluster-ops ticket; choose a registration tool for the DBM baseline (or defer it, science plan §5.4); choose the invariance-layer filter family (architecture §6 item 5); confirm CNN-training label scope (architecture §6 item 4) | nothing | Phase 3 |

0a/0b/0c/0d have no dependencies on each other — run all four simultaneously.

### Phase 1 — SimCLR mechanism experiment (starts once 0a completes)

Per `simclr_experimental_plan.md` §5-§6: `SimCLR(pretrained=True)` vs. `SimCLR(pretrained=False)`, MCA-perturbed at inference, on NKI `Longitudinal_Adult` zero-change pairs. ~3 days pilot (n=20) once Gate 0 passes, +1 more if the running-statistics fix (simclr_plan §2.2) is needed. This is independent of Phases 2-3 and produces a result on the core "does training amplify numerical sensitivity" question before the heavier geometric infrastructure is even built.

### Phase 2 — Invariance-layer validation (starts once 0b + 0c complete)

Per `geometric_longitudinal_validation_architecture.md` §2.2-§2.3: build the band-limit invariance layer and per-subject self-calibration, calibrate the filter cutoff on the pilot split only, then run the raw-vs-invariant comparison on **Experiment 1 (zero-change calibration) alone** as a standalone gate before wiring into anything else. Architecture doc's own criterion: invariant must improve calibration margin over raw without losing >0.05 dose-response AUROC (architecture §5 config). If it fails this gate, the added complexity isn't justified — report that, don't adopt it silently, and Phase 3 proceeds with `raw` only.

### Phase 3 — Full confirmatory grid (starts once Phase 2 gate resolves + 0d decisions land)

Science plan Experiments 1-6, run for both `raw` and `invariant` CNN variants (if Phase 2 passed) × 2 cohorts × 2 architectures × 5 seeds, per the 40-cell reduced design (science plan §7). FreeSurfer cross-sectional + longitudinal stream as a CPU array (blocked on 0d's cluster-ops ticket — this is the plan's dominant runtime driver: ~3 `recon-all` runs/subject × 8-14h each, per architecture doc §6). DBM baseline conditional on 0d's registration-tool decision.

**Open gap, not resolved by any of the three documents:** Phase 3's trained-CNN comparator (science plan §5.2) has no source. SimCLR (Phase 1) is scoped to the mechanism question on NKI-adult zero-change pairs only — it was never wired into the full Jacobian-ground-truth grid, and doing so would hit the same Track A preprocessing asymmetry documented in `simclr_experimental_plan.md` §4.1 (SimCLR needs its own TurboPrep-to-MNI152 preprocessing, incompatible with the deformation-ground-truth Track A design without risking the CAT12-style registration-absorbs-the-synthetic-deformation problem flagged earlier in this review). **Recommendation: run Phase 3 initially with `un-CNN (raw/invariant)` vs. `FreeSurfer` vs. `DBM baseline` only — no trained-CNN arm — and treat Phase 1's SimCLR result as a separate, clearly-labeled finding rather than forcing a premature merge.** Revisit a Phase-3-compatible trained comparator only after Phase 1 lands and `architecture/geometric/trained_cnn.py` (build order step 6, explicitly sequenced last) is scoped.

## 4. Unified timeline

| Day | Phase 0 (parallel) | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| 1 | 0a/0b/0c start; 0d tickets filed and decisions made | — | — | — |
| 2 | 0a likely complete (Gate 0 + zero-change pairs) | starts | — | — |
| 3-5 | 0c continues (deformation+QC, 3-5 days); 0b likely complete | pilot run, gates computed | starts once 0b+0c done | — |
| 5-7 | 0c complete | expand to confirmatory sample if pilot is decisive; **first result available** | invariance layer built, pilot-calibrated, Experiment-1 gate run | — |
| 7-9 | — | done | gate resolves: adopt `invariant` or fall back to `raw` | starts if 0d resolved |
| 9-16+ | — | — | — | full grid; FreeSurfer throughput is the dominant uncertainty (science plan §9 already hedges "3-7 days, subject to measured FreeSurfer throughput" — that estimate did not account for Phase 2 or the 0d lead time, so 9-16 days is more realistic end-to-end) |

Phase 1's result lands around Day 5-7 regardless of how Phase 3 goes — it does not need to wait for FreeSurfer, the deformation module, or any 0d decision.

## 5. Immediate next actions (Day 1 checklist)

1. File the FreeSurfer cluster-ops ticket (0d) — longest lead time item, start it first even though it's "just a decision."
2. Start `simclr_experimental_plan.md` §2 Gate 0 implementation (0a).
3. Start `geometric/determinism.py` (0b) and `geometric/deformation.py`/`qc.py` (0c) in parallel — both pure-Python, no external blockers.
4. Decide registration tool for the DBM baseline (0d) — SimpleITK vs. ANTsPy vs. ANTs CLI module vs. defer-as-conditional-followup (science plan already permits deferring this baseline).
5. Confirm invariance-layer filter family (0d) — architecture doc's open question 5: generic Gaussian low-pass vs. a filter shaped to `taper_boundary_voxels`.

None of these five block each other. All five can start today.
