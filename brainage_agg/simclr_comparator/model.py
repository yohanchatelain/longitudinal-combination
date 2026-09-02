"""SimCLR 3D ResNet-18 backbone loader, trained or randomly-initialized.

The upstream model (Kaczmarek et al., github.com/emilykaczmarek/3D-Neuro-SimCLR) builds
its encoder via a thin wrapper (`simclr/modules/resnet.py::get_resnet`) around MONAI's
own `monai.networks.nets.resnet18` — confirmed by inspecting the cloned repo: the wrapper
file is a near-verbatim copy of MONAI's ResNet (same class names, same
Apache-2.0-licensed implementation). Rather than vendor that copy, this module depends on
MONAI directly and calls the same underlying architecture, which is functionally
identical and keeps the license/attribution and any upstream bugfixes with MONAI itself.

Training-time wrapper (simclr/simclr/simclr.py, confirmed in the cloned repo):

    class SimCLR(nn.Module):
        def __init__(self, encoder, projection_dim, n_features):
            self.encoder = encoder
            self.encoder.fc = Identity()          # pooled 512-d features, no classifier
            self.projector = nn.Sequential(...)     # discarded here — Gate 0 and the
                                                       # mechanism experiment both want
                                                       # h_i = encoder(x), not z_i.

n_features=512 and projection_dim=64 are read directly from the cloned repo's
`config/config.yaml` (`resnet: "resnet18"`, `projection_dim: 64`) — 512 is also
architecturally fixed for resnet18 (block_inplanes[3] * BasicBlock.expansion = 512 * 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import resnet18

CHECKPOINT_URL = (
    "https://github.com/emilykaczmarek/3D-Neuro-SimCLR/releases/download/"
    "v1.0.0/simclr_3d_brain_foundation.tar"
)
N_FEATURES = 512


def _build_encoder() -> nn.Module:
    """Construct the bare 3D ResNet-18 encoder, fc stripped to Identity.

    `feed_forward=False` avoids MONAI's `nn.Linear(512, num_classes)` head entirely
    (the upstream `get_resnet(..., num_classes=0)` call would otherwise construct a
    degenerate `nn.Linear(512, 0)` and immediately overwrite it with Identity() in the
    SimCLR wrapper — skipping straight to `feed_forward=False` reaches the same result
    without ever building the degenerate layer).
    """
    return resnet18(spatial_dims=3, n_input_channels=1, num_classes=0, feed_forward=False)


def load_simclr_backbone(*, pretrained: bool, device: str = "cpu") -> nn.Module:
    """Load the SimCLR encoder — trained checkpoint or PyTorch-default random init.

    pretrained=True: downloads and loads the released checkpoint. **Unverified without
    network access**: the exact top-level structure of the downloaded `.tar` (a raw
    `state_dict`, or a dict with a `"model"`/`"state_dict"` key, per the common
    convention also seen in AnatCL's checkpoint format) is not confirmed from the
    repository's README alone. This function tries the raw-state-dict path first, then
    falls back to common wrapper-key names, and raises with a clear message listing the
    top-level keys actually found if neither matches — do not silently swallow a
    structure mismatch. This is the "one-time network-dependent smoke check" called out
    as still-needed in the approved implementation plan; run it before trusting this
    path in Gate 0.

    pretrained=False: PyTorch's default init (Kaiming/uniform per-layer defaults,
    `BatchNorm3d` running stats at their untrained defaults of mean 0 / var 1) — this is
    exactly the object `gate0.run_gate0` exists to validate, not a stand-in for it.
    """
    encoder = _build_encoder()

    if not pretrained:
        return encoder.to(device).eval()

    checkpoint = torch.hub.load_state_dict_from_url(
        CHECKPOINT_URL, map_location="cpu", weights_only=False
    )

    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Unexpected checkpoint structure: top-level type {type(checkpoint)!r}. "
            "Expected a state_dict or a dict wrapping one under 'model'/'state_dict'."
        )

    encoder_prefix = "encoder."
    encoder_state = {
        key[len(encoder_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(encoder_prefix)
    }
    if not encoder_state:
        # Checkpoint may already be encoder-only (no "encoder." prefix) — try as-is
        # before giving up, but fail loudly rather than loading a silently-empty model.
        encoder_state = state_dict

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "SimCLR checkpoint did not load cleanly into the encoder — "
            f"missing keys: {missing}, unexpected keys: {unexpected}. "
            f"Top-level checkpoint keys were: {list(state_dict.keys())[:10]}... "
            "Checkpoint structure assumption in load_simclr_backbone() needs updating."
        )

    return encoder.to(device).eval()
