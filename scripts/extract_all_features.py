#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import FreeSurferFeatureDataset
from src.io import load_cohort
from src.models import CNN3D_CovPool, CNN3D_DoubleConv, freeze_model, init_xavier
from src.utils import ensure_dir, load_config, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug sample limit.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument(
        "--seed-batch-size",
        type=int,
        default=0,
        help="Number of CNN seeds to extract per preprocessing pass. Default extracts all pending seeds together.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help='CUDA device ids to use, e.g. "0", "0,1,2", or "all". Ignored unless cnn.device is cuda.',
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override cnn.num_workers for multiprocessing data loading/preprocessing.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches each DataLoader worker prefetches when num_workers > 0.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader worker processes alive across epochs/configurations.",
    )
    parser.add_argument(
        "--model",
        default="double_conv",
        choices=["double_conv", "cov_pool"],
        help="CNN architecture to use for feature extraction.",
    )
    return parser.parse_args()


def model_device(model):
    return next(model.parameters()).device


def extract_features_for_models(models, loader):
    features = {seed: [] for seed in models}
    order = []
    devices = sorted({model_device(model) for model in models.values()}, key=str)
    with torch.no_grad():
        for x, idx in tqdm(loader, desc=f"extract {len(models)} seed(s)", leave=False):
            order.extend(np.asarray(idx).tolist())
            batches = {device: x.to(device, non_blocking=True) for device in devices}
            outputs = []
            for seed, model in models.items():
                outputs.append((seed, model(batches[model_device(model)])))
            for seed, output in outputs:
                features[seed].append(output.cpu().numpy())
    sort_idx = np.argsort(np.asarray(order))
    return {seed: np.concatenate(parts, axis=0)[sort_idx] for seed, parts in features.items()}


def chunked(values, chunk_size):
    values = list(values)
    if chunk_size <= 0:
        yield values
        return
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


_MODEL_CLASSES = {
    "double_conv": CNN3D_DoubleConv,
    "cov_pool": CNN3D_CovPool,
}

_MODEL_JSON_NAMES = {
    "double_conv": "CNN3D_DoubleConv",
    "cov_pool": "CNN3D_CovPool",
}


def build_model(seed, channels, cnn_cfg, device, cnn_arch="double_conv"):
    set_seed(int(seed))
    cls = _MODEL_CLASSES[cnn_arch]
    if cnn_arch == "double_conv":
        model = cls(
            in_channels=len(channels),
            use_norm=cnn_cfg.get("use_norm", True),
            activation=cnn_cfg.get("activation", "leaky_relu"),
        )
    else:
        model = cls(in_channels=len(channels))
    if cnn_cfg.get("init", "xavier") == "xavier":
        init_xavier(model)
    freeze_model(model)
    return model.to(device).eval()


def output_paths(out_dir, scaler, channel_name, seed, cnn_arch="double_conv"):
    if cnn_arch == "double_conv":
        stem = f"features__scaler-{scaler}__channels-{channel_name}__seed-{seed}"
    else:
        stem = f"features__model-{cnn_arch}__scaler-{scaler}__channels-{channel_name}__seed-{seed}"
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.json"


def save_feature_file(out_path, json_path, X, df, scaler, channels, seed, cnn_cfg, cnn_arch="double_conv"):
    np.savez_compressed(
        out_path,
        X=X,
        **metadata_arrays(df),
        scaler=scaler,
        channels="_".join(channels),
        seed=int(seed),
        cnn_arch=cnn_arch,
    )
    write_json(
        json_path,
        {
            "scaler": scaler,
            "channels": channels,
            "seed": int(seed),
            "cnn_arch": cnn_arch,
            "model": _MODEL_JSON_NAMES[cnn_arch],
            "use_norm": bool(cnn_cfg.get("use_norm", True)) if cnn_arch == "double_conv" else None,
            "activation": cnn_cfg.get("activation", "leaky_relu") if cnn_arch == "double_conv" else None,
            "n_samples": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
        },
    )


def metadata_arrays(df):
    keys = [
        "id",
        "participant_id",
        "session",
        "session_type",
        "age",
        "sex",
        "timepoint",
        "directory",
        "days",
        "days_since_enrollment",
        "studies",
        "study_num",
        "image_path",
        "mask_path",
    ]
    return {key: df[key].values for key in keys if key in df.columns}


def resolve_device(requested_device: str) -> torch.device:
    if requested_device != "cuda":
        return torch.device(requested_device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cuda_available = torch.cuda.is_available()

    if cuda_available:
        return torch.device("cuda")

    print("Warning: CUDA requested but unavailable; using CPU.")
    return torch.device("cpu")


def resolve_devices(requested_device: str, gpu_arg: str) -> list[torch.device]:
    device = resolve_device(requested_device)
    if device.type != "cuda":
        return [device]

    count = torch.cuda.device_count()
    if gpu_arg == "all":
        ids = list(range(count))
    else:
        ids = [int(part) for part in gpu_arg.split(",") if part.strip()]

    invalid = [idx for idx in ids if idx < 0 or idx >= count]
    if invalid:
        raise ValueError(f"Invalid CUDA device ids {invalid}; visible device count is {count}")
    if not ids:
        raise ValueError("No CUDA device ids selected.")
    return [torch.device(f"cuda:{idx}") for idx in ids]


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    out_dir = ensure_dir(config["output"]["feature_dir"])

    df = load_cohort(
        data_cfg["cohort_csv"],
        freesurfer_root=data_cfg.get("freesurfer_root"),
        image_file=data_cfg.get("image_file", "mri/brain.mgz"),
        mask_file=data_cfg.get("mask_file", "mri/brainmask.mgz"),
        verbose=args.verbose,
        use_csv_paths=data_cfg.get("use_csv_paths", False),
    )
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No rows with existing image and mask files.")

    cnn_cfg = config["cnn"]
    prep_cfg = config["preprocessing"]
    cnn_arch = args.model
    devices = resolve_devices(cnn_cfg.get("device", "cuda"), args.gpus)
    num_workers = (
        args.num_workers if args.num_workers is not None else cnn_cfg.get("num_workers", 0)
    )
    pin_memory = any(device.type == "cuda" for device in devices)
    print(f"Using devices: {', '.join(str(device) for device in devices)}")
    print(f"Using DataLoader workers: {num_workers}")

    for scaler in config["scalers"]:
        for feature_set in config["feature_sets"]:
            channels = feature_set["channels"]
            channel_name = "_".join(channels)
            dataset = FreeSurferFeatureDataset(
                df,
                scaler=scaler,
                channels=channels,
                clip_low=prep_cfg.get("clip_low", 1),
                clip_high=prep_cfg.get("clip_high", 99),
                rank_size=prep_cfg.get("rank_size", 3),
            )
            loader_kwargs = {
                "batch_size": cnn_cfg.get("batch_size", 1),
                "num_workers": num_workers,
                "shuffle": False,
                "pin_memory": pin_memory,
            }
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = args.prefetch_factor
                loader_kwargs["persistent_workers"] = args.persistent_workers
            loader = DataLoader(dataset, **loader_kwargs)
            pending_seeds = []
            for seed in config["seeds"]:
                out_path, _ = output_paths(out_dir, scaler, channel_name, seed, cnn_arch)
                if out_path.exists() and not args.overwrite:
                    print(f"Skipping existing {out_path}")
                    continue
                pending_seeds.append(seed)

            for seed_chunk in chunked(pending_seeds, args.seed_batch_size):
                if not seed_chunk:
                    continue
                models = {
                    seed: build_model(seed, channels, cnn_cfg, devices[pos % len(devices)], cnn_arch)
                    for pos, seed in enumerate(seed_chunk)
                }
                print(
                    f"Extracting model={cnn_arch} scaler={scaler} channels={channel_name} "
                    f"seeds={list(seed_chunk)} on {', '.join(str(device) for device in devices)}"
                )
                features_by_seed = extract_features_for_models(models, loader)
                for seed, X in features_by_seed.items():
                    out_path, json_path = output_paths(out_dir, scaler, channel_name, seed, cnn_arch)
                    save_feature_file(out_path, json_path, X, df, scaler, channels, seed, cnn_cfg, cnn_arch)
                del models
                if any(device.type == "cuda" for device in devices):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
