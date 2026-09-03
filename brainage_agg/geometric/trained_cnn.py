"""Supervised training for the trained-CNN comparator (plan §5.2).

Confirmed absent anywhere else in this repo: every CNN here today is frozen-random
(`run_voxel_importance._build_cnn` Xavier-inits a `CNN3D_DoubleConv`/`CNN3D_CovPool`
then freezes it). The trained comparator uses the identical architecture and
preprocessing as that frozen arm -- only the learned weights differ (plan §5.2) -- so
optimization's contribution to nuisance sensitivity can be isolated from architecture
and preprocessing choices.

`CNN3D_DoubleConv`/`CNN3D_CovPool` are feature extractors (they output a flattened
multiscale feature vector, not a scalar), so training wraps the backbone with a
throwaway linear regression head to get a training signal; `load_trained_checkpoint`
returns the bare, frozen backbone afterward, matching exactly how the frozen arm
exposes a bare backbone downstream (voxel attribution, feature-change extraction, ...
all operate on backbone features, never on a head's scalar prediction).

Plan §5.2 constraints enforced here, not just documented:
- never trained/fine-tuned on the synthetic geometric-validation transformations
  (this module only ever sees real cohort images the caller supplies);
- no subject overlap between training and the geometric-validation pilot/validation
  splits -- checked and raised on, not assumed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models import CNN3D_CovPool, CNN3D_DoubleConv, freeze_model

from .determinism import enforce_deterministic_environment


@dataclass(frozen=True)
class TrainingExample:
    subject_id: str
    channels: np.ndarray  # (C, D, H, W) float32, already preprocessed
    label: float  # e.g. age


class SubjectVolumeDataset(Dataset):
    """A minimal, in-memory dataset of preprocessed per-visit channel volumes.

    Loading/preprocessing (`build_channels`, cohort I/O) is the orchestrator's job;
    this dataset only wraps already-built `TrainingExample`s so the training-loop
    logic below can be unit-tested with no cohort/FreeSurfer dependency.
    """

    def __init__(self, examples: Sequence[TrainingExample]):
        if not examples:
            raise ValueError("SubjectVolumeDataset requires at least one example")
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self._examples[index]
        return (
            torch.as_tensor(example.channels, dtype=torch.float32),
            torch.as_tensor(example.label, dtype=torch.float32),
        )


def build_cnn(architecture: str, in_channels: int) -> nn.Module:
    """The trainable-arm backbone factory -- the counterpart to
    `run_voxel_importance._build_cnn`'s frozen-random factory. Uses the framework's
    default initialization rather than Xavier re-init, since this backbone is about
    to be optimized rather than used as a fixed random feature extractor.
    """
    if architecture == "double_conv":
        return CNN3D_DoubleConv(in_channels=in_channels, use_norm=True, activation="leaky_relu")
    if architecture == "cov_pool":
        return CNN3D_CovPool(in_channels=in_channels)
    raise ValueError(f"Unknown arch: {architecture!r}. Choose 'double_conv' or 'cov_pool'.")


class _BackboneWithHead(nn.Module):
    """Backbone + linear regression head, for supervised training only.

    Downstream geometric-validation analysis always uses the backbone alone (see
    `load_trained_checkpoint`); the head exists only to give the backbone's weights
    a training signal.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


def assert_no_training_subject_overlap(
    training_subject_ids: Iterable[str],
    held_out_subject_ids: Iterable[str],
) -> None:
    """Plan §5.2: 'Ensure no subject overlap between model training and validation.'

    A hard failure, not a convention: raised before any training happens.
    """
    overlap = set(map(str, training_subject_ids)) & set(map(str, held_out_subject_ids))
    if overlap:
        raise ValueError(
            f"{len(overlap)} subject(s) appear in both the training set and the "
            f"held-out split: {sorted(overlap)[:10]}"
        )


@dataclass(frozen=True)
class TrainingHistoryEpoch:
    epoch: int
    train_loss: float
    val_loss: float


@dataclass(frozen=True)
class TrainingResult:
    architecture: str
    in_channels: int
    seed: int
    label_name: str
    best_epoch: int
    best_val_loss: float
    history: list[TrainingHistoryEpoch] = field(default_factory=list)
    backbone_state_dict: dict = field(default_factory=dict)


def train_cnn(
    architecture: str,
    train_examples: Sequence[TrainingExample],
    val_examples: Sequence[TrainingExample],
    *,
    held_out_subject_ids: Iterable[str] = (),
    label_name: str = "age",
    seed: int = 0,
    max_epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    device: str = "cpu",
) -> TrainingResult:
    """Train one backbone checkpoint for the trained-CNN comparator.

    Early-stops on validation MAE after `patience` epochs with no improvement and
    returns the *best*-epoch backbone weights, not necessarily the final epoch's, so
    a checkpoint never reflects an overfit tail of training. `held_out_subject_ids`
    should be the geometric-validation pilot + validation subject IDs (plan §5.2);
    passing it is how the no-overlap constraint gets enforced, not merely documented.
    """
    training_ids = {example.subject_id for example in train_examples}
    validation_ids = {example.subject_id for example in val_examples}
    assert_no_training_subject_overlap(training_ids, validation_ids)
    assert_no_training_subject_overlap(training_ids | validation_ids, held_out_subject_ids)

    enforce_deterministic_environment(seed)
    torch.manual_seed(seed)
    in_channels = train_examples[0].channels.shape[0]
    backbone = build_cnn(architecture, in_channels).to(device)
    model = _BackboneWithHead(backbone, backbone.out_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    train_loader = DataLoader(SubjectVolumeDataset(train_examples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SubjectVolumeDataset(val_examples), batch_size=batch_size, shuffle=False)

    history: list[TrainingHistoryEpoch] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for channels, labels in train_loader:
            channels, labels = channels.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(channels)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for channels, labels in val_loader:
                channels, labels = channels.to(device), labels.to(device)
                val_losses.append(float(loss_fn(model(channels), labels)))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append(TrainingHistoryEpoch(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training completed without ever recording a best checkpoint")
    return TrainingResult(
        architecture=architecture,
        in_channels=in_channels,
        seed=seed,
        label_name=label_name,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        history=history,
        backbone_state_dict=best_state,
    )


def save_checkpoint(result: TrainingResult, checkpoint_dir: Path) -> Path:
    """Persist a trained backbone checkpoint plus its training provenance to disk."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{result.architecture}__{result.label_name}__seed_{result.seed}.pt"
    torch.save(
        {
            "architecture": result.architecture,
            "in_channels": result.in_channels,
            "seed": result.seed,
            "label_name": result.label_name,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "history": [vars(epoch) for epoch in result.history],
            "backbone_state_dict": result.backbone_state_dict,
        },
        path,
    )
    return path


def load_trained_checkpoint(checkpoint_path: Path, *, device: str = "cpu") -> nn.Module:
    """Load a trained, frozen backbone -- the trained-arm counterpart to
    `run_voxel_importance._build_cnn`'s frozen-random factory. Returns a bare
    `CNN3D_DoubleConv`/`CNN3D_CovPool` instance in eval mode with gradients disabled,
    so downstream code (feature-change extraction, attribution) can treat the raw
    and trained arms identically -- only the weights and their provenance differ.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone = build_cnn(payload["architecture"], payload["in_channels"])
    backbone.load_state_dict(payload["backbone_state_dict"])
    freeze_model(backbone)
    return backbone.to(device).eval()
