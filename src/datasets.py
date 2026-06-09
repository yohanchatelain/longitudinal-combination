from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .io import load_mgz_float32, load_mgz_mask
from .preprocessing import build_channels


class FreeSurferFeatureDataset(Dataset):
    def __init__(
        self,
        dataframe,
        scaler,
        channels,
        clip_low=1,
        clip_high=99,
        rank_size=3,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.scaler = scaler
        self.channels = list(channels)
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.rank_size = rank_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img = load_mgz_float32(row["image_path"])
        mask_path = row.get("mask_path", "") if hasattr(row, "get") else row["mask_path"]
        if mask_path:
            mask = load_mgz_mask(mask_path)
        else:
            mask = img > 0  # fallback: non-zero voxels (for raw T1w without explicit mask)
        x = build_channels(
            img,
            mask,
            self.scaler,
            self.channels,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            rank_size=self.rank_size,
        )
        return torch.from_numpy(x).float(), idx
