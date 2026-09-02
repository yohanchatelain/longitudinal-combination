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

**Checkpoint structure — confirmed by downloading and inspecting the actual released
file** (previously only guessed at from the README): top level is
`{"epoch": int, "model_state_dict": {...}, "optimizer_state_dict": {...}}`, and encoder
weights inside `model_state_dict` are prefixed `module.encoder.` (the `module.` prefix
comes from `nn.DataParallel`/`DistributedDataParallel` wrapping the model during
training — a very common training-checkpoint artifact, not anything specific to this
release).
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

    pretrained=True: downloads and loads the released checkpoint (confirmed structure,
    see module docstring). Uses `weights_only=True` — this checkpoint is a plain dict of
    tensors/ints, so full unpickling is not needed, and restricting to the safe
    tensor-loading path avoids executing arbitrary pickled objects from a downloaded
    file. Strips the `module.encoder.` prefix (DataParallel wrapping + the SimCLR
    wrapper's own `encoder` attribute name) and raises with the actual keys found if the
    structure ever changes upstream — do not silently swallow a mismatch.

    pretrained=False: PyTorch's default init (Kaiming/uniform per-layer defaults,
    `BatchNorm3d` running stats at their untrained defaults of mean 0 / var 1) — this is
    exactly the object `gate0.run_gate0` exists to validate, not a stand-in for it.
    """
    encoder = _build_encoder()

    if not pretrained:
        return encoder.to(device).eval()

    checkpoint = torch.hub.load_state_dict_from_url(
        CHECKPOINT_URL, map_location="cpu", weights_only=True
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unexpected checkpoint structure: top-level keys "
            f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)!r}. "
            "Expected a dict with a 'model_state_dict' key."
        )
    state_dict = checkpoint["model_state_dict"]

    encoder_prefix = "module.encoder."
    encoder_state = {
        key[len(encoder_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(encoder_prefix)
    }
    if not encoder_state:
        raise ValueError(
            f"No keys with prefix {encoder_prefix!r} found in model_state_dict. "
            f"First 10 keys were: {list(state_dict.keys())[:10]}. "
            "Checkpoint structure assumption in load_simclr_backbone() needs updating."
        )

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "SimCLR checkpoint did not load cleanly into the encoder — "
            f"missing keys: {missing}, unexpected keys: {unexpected}."
        )

    return encoder.to(device).eval()
