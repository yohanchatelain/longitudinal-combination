"""Phase 4a pilot: compare two permutation-null designs for architecture-bias
correction of voxel attribution maps (`w_t` only).

  index_perm  -- cheap. Cache per-feature gradients (`cache_topk_gradients`,
                 k features), then for N permutations of `w_t_std`, recombine
                 via einsum (no extra backward passes).
  group_perm  -- expensive but exact. For N permutations of the group labels,
                 recompute `w_t`, `rank_normalize`, and run a full
                 `backprop_from_w` per permutation.

Scope: 1 dataset x 1 aggregation x n_seeds x n_subjects (NKI/mean, 2 seeds, 5
subjects by default -- the Phase 4a pilot scale). Compares the two null
*mean* region tables (Pearson r on mean_abs_importance / mean_signed_importance)
and reports wall-clock per design, to decide the null design for the Phase 4b
full sweep.

Usage (submit via sbatch -- this runs forward_cache/backprop_from_w on real
MRI data):
    python -m brainage_agg.experiment.run_permutation_null_pilot \\
        --dataset nki --agg mean --n-subjects 5 --n-seeds 2 \\
        --topk 50 --n-permutations 20 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.analysis.voxel_attribution import (
    backprop_from_w,
    forward_cache,
    load_freesurfer_lut,
    project_to_atlas,
)
from brainage_agg.analysis.importance_weights import compute_w_t, rank_normalize
from brainage_agg.analysis.permutation_null import (
    WelfordAccumulator,
    cache_topk_gradients,
    topk_feature_indices,
)
from brainage_agg.analysis.group_diff import _get_dataset_config
from brainage_agg.data.manifest import load_manifest
from brainage_agg.experiment.run_voxel_importance import (
    _build_aggregated_features,
    _build_cnn,
    _find_npz_files,
    _preprocess_all_subjects,
    _region_importance_df,
    _sample_subjects,
)
from src.io import load_cohort


def _upsample(arr_small: np.ndarray, original_size: tuple[int, int, int]) -> np.ndarray:
    """Trilinearly upsample a (h,w,d) array to (H,W,D), matching backprop_from_w."""
    t = torch.from_numpy(arr_small).float()[None, None]
    up = F.interpolate(t, size=original_size, mode="trilinear", align_corners=False)
    return up[0, 0].numpy().astype(np.float32)


def run(
    project_root: Path,
    dataset: str,
    arch: str,
    scaler: str,
    channels: str,
    aggregation: str,
    n_subjects: int,
    n_seeds: int,
    topk: int,
    n_permutations: int,
    device: str,
    downsample_factor: int,
    rng_seed: int,
) -> None:
    ds_cfg = _get_dataset_config(dataset, project_root)
    groups: tuple[str, str] = ds_cfg["groups"]
    result_dir = Path(ds_cfg["result_dir"])
    feature_dir = Path(ds_cfg["feature_dir"])

    manifest = load_manifest(result_dir / "manifest.csv")

    npz_files = _find_npz_files(feature_dir, arch, scaler, channels)[:n_seeds]
    if not npz_files:
        raise FileNotFoundError(
            f"No NPZ files found for arch={arch}, scaler={scaler}, channels={channels} "
            f"in {feature_dir}"
        )

    cfg_path = project_root / "brainage_agg" / (
        "ppmi_config.yaml" if dataset == "ppmi" else "config.yaml"
    )
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    cohort_csv = project_root / cfg["data"]["cohort_csv"]
    fs_root_cfg = cfg["data"].get("fs_root")
    fs_root = str(project_root / fs_root_cfg) if fs_root_cfg else None
    if fs_root is None:
        raise RuntimeError("Phase 4a pilot requires fs_root (for aparc+aseg parcellations).")

    cohort = load_cohort(str(cohort_csv), freesurfer_root=fs_root)
    sampled = _sample_subjects(cohort, groups, n_subjects, manifest)
    print(f"Sampled {len(sampled)} subjects for the pilot")

    lut = load_freesurfer_lut(project_root / "FS-LUT.txt")

    channels_list = channels.split("_")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Preprocessing {len(sampled)} subjects (n_jobs={n_jobs})...")
    t0 = time.time()
    cached_subjects = _preprocess_all_subjects(
        sampled, scaler, channels_list, fs_root, n_jobs, "aparc+aseg.mgz"
    )
    print(f"Done in {time.time() - t0:.0f}s -- {len(cached_subjects)}/{len(sampled)} valid")
    if not cached_subjects:
        print("No valid subjects -- aborting.")
        return

    cached_tensors = [
        (torch.tensor(arr[np.newaxis], dtype=torch.float32), parc, arr.any(axis=0))
        for arr, parc, _affine in cached_subjects
    ]
    del cached_subjects

    region_signed_accum: dict[str, dict[int, list[float]]] = {"index_perm": {}, "group_perm": {}}
    region_abs_accum: dict[str, dict[int, list[float]]] = {"index_perm": {}, "group_perm": {}}
    region_nvox: dict[str, dict[int, int]] = {"index_perm": {}, "group_perm": {}}
    idx_times: list[float] = []
    grp_times: list[float] = []
    n_seeds_used = 0

    for npz_path in npz_files:
        agg_result = _build_aggregated_features(npz_path, manifest, aggregation, groups)
        if agg_result is None:
            print(f"  {npz_path.name}: skipped (insufficient data)")
            continue
        X_agg, bands_agg, seed = agg_result
        n_seeds_used += 1
        print(f"\nSeed {seed}: X_agg={X_agg.shape}")

        w_t_std = rank_normalize(compute_w_t(X_agg, bands_agg, groups))
        topk_idx = topk_feature_indices({"tstat": w_t_std}, k=topk)
        print(f"  topk_idx: {len(topk_idx)} unique features (k={topk})")

        rng = np.random.default_rng(rng_seed + seed)

        # Group-label permutations don't depend on the per-subject cache --
        # precompute once per seed and reuse for every sampled subject.
        t0 = time.time()
        group_perm_weights = [
            rank_normalize(compute_w_t(X_agg, rng.permutation(bands_agg), groups))
            for _ in range(n_permutations)
        ]
        print(f"  Precomputed {n_permutations} group-label-permutation w_t vectors "
              f"in {time.time() - t0:.1f}s")

        model = _build_cnn(arch, channels_list, seed, device)
        n_model_feat = getattr(model, "out_features", None)
        assert n_model_feat is not None, "Model has no out_features attribute"

        for x_tensor, parc, brain_mask in cached_tensors:
            cache = forward_cache(model, x_tensor, device=device, downsample_factor=downsample_factor)

            # --- index_perm null ---
            t0 = time.time()
            G = cache_topk_gradients(cache, n_model_feat, topk_idx, retain_graph=True)
            spatial_shape = G.shape[2:]  # (h, w, d)
            acc_idx_signed = WelfordAccumulator(spatial_shape)
            acc_idx_abs = WelfordAccumulator(spatial_shape)
            for _ in range(n_permutations):
                w_perm = rng.permutation(w_t_std)
                w_sub = w_perm[topk_idx].astype(np.float32)
                S = np.einsum("k,kc...->c...", w_sub, G)
                acc_idx_signed.update(S.mean(axis=0))
                acc_idx_abs.update(np.abs(S).mean(axis=0))
            idx_times.append(time.time() - t0)

            # --- group_perm null ---
            t0 = time.time()
            acc_grp_signed = WelfordAccumulator(cache.original_size)
            acc_grp_abs = WelfordAccumulator(cache.original_size)
            for i, w_perm in enumerate(group_perm_weights):
                last = i == len(group_perm_weights) - 1
                signed_map, abs_map = backprop_from_w(
                    cache, w_perm, n_model_feat, retain_graph=not last
                )
                acc_grp_signed.update(signed_map)
                acc_grp_abs.update(abs_map)
            grp_times.append(time.time() - t0)

            # --- finalize, mask, project ---
            idx_mean_signed = _upsample(acc_idx_signed.finalize()[0], cache.original_size)
            idx_mean_abs = _upsample(acc_idx_abs.finalize()[0], cache.original_size)
            grp_mean_signed, _ = acc_grp_signed.finalize()
            grp_mean_abs, _ = acc_grp_abs.finalize()

            for m in (idx_mean_signed, idx_mean_abs, grp_mean_signed, grp_mean_abs):
                m[~brain_mask] = 0.0

            for tag, signed_m, abs_m in [
                ("index_perm", idx_mean_signed, idx_mean_abs),
                ("group_perm", grp_mean_signed, grp_mean_abs),
            ]:
                reg_signed, reg_abs = project_to_atlas(signed_m, abs_m, parc)
                for lid, (val, n_vox) in reg_abs.items():
                    region_abs_accum[tag].setdefault(lid, []).append(val)
                    region_nvox[tag][lid] = n_vox
                for lid, (val, _) in reg_signed.items():
                    region_signed_accum[tag].setdefault(lid, []).append(val)

        del model

    if n_seeds_used == 0:
        print("No valid seeds -- aborting.")
        return

    idx_df = _region_importance_df(
        region_signed_accum["index_perm"], region_abs_accum["index_perm"],
        region_nvox["index_perm"], lut, groups,
    )
    grp_df = _region_importance_df(
        region_signed_accum["group_perm"], region_abs_accum["group_perm"],
        region_nvox["group_perm"], lut, groups,
    )

    merged = pd.merge(
        idx_df[["label_id", "region_name", "mean_abs_importance", "mean_signed_importance"]],
        grp_df[["label_id", "region_name", "mean_abs_importance", "mean_signed_importance"]],
        on=["label_id", "region_name"], suffixes=("_index", "_group"),
    )
    r_abs = merged["mean_abs_importance_index"].corr(merged["mean_abs_importance_group"])
    r_signed = merged["mean_signed_importance_index"].corr(merged["mean_signed_importance_group"])

    out_dir = result_dir / "group_diff" / "voxel_importance" / "permutation_null_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{arch}__{scaler}__{channels}__{aggregation}__nsub{len(cached_tensors)}__{n_seeds_used}seeds__topk{topk}__nperm{n_permutations}"
    csv_path = out_dir / f"null_comparison_{tag}.csv"
    merged.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("Phase 4a pilot: index_perm vs group_perm null comparison")
    print(f"{'='*60}")
    print(f"  n_seeds={n_seeds_used}, n_subjects={len(cached_tensors)}, "
          f"topk={topk}, n_permutations={n_permutations}")
    print(f"  index_perm: {np.mean(idx_times):.2f}s/subject "
          f"(cache_topk_gradients [{topk} backward passes, no upsample] + recombine)")
    print(f"  group_perm: {np.mean(grp_times):.2f}s/subject "
          f"({n_permutations} full backward passes, with upsample)")
    print(f"  Pearson r (mean_abs_importance, {len(merged)} regions): {r_abs:.3f}")
    print(f"  Pearson r (mean_signed_importance, {len(merged)} regions): {r_signed:.3f}")
    print(f"  Saved {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["nki", "ppmi"], default="nki")
    parser.add_argument("--arch", default="double_conv", choices=["double_conv", "cov_pool"])
    parser.add_argument("--scaler", default="minmax")
    parser.add_argument("--channels", default="t1_rank_sobel")
    parser.add_argument("--agg", default="mean", help="Single aggregation to pilot. Default: mean.")
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--topk", type=int, default=50,
                         help="Number of top-|w_t| features to cache gradients for "
                              "(index_perm null). Default: 50.")
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    run(
        project_root=ROOT,
        dataset=args.dataset,
        arch=args.arch,
        scaler=args.scaler,
        channels=args.channels,
        aggregation=args.agg,
        n_subjects=args.n_subjects,
        n_seeds=args.n_seeds,
        topk=args.topk,
        n_permutations=args.n_permutations,
        device=args.device,
        downsample_factor=args.downsample,
        rng_seed=args.rng_seed,
    )


if __name__ == "__main__":
    main()
