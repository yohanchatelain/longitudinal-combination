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
file**: top level is `{"epoch": int, "model_state_dict": {...}, "optimizer_state_dict":
{...}}`, and encoder weights inside `model_state_dict` are prefixed `module.encoder.`
(the `module.` prefix comes from `nn.DataParallel`/`DistributedDataParallel` wrapping the
model during training — a common training-checkpoint artifact, not specific to this
release). `CHECKPOINT_SHA256` below is the hash of that confirmed download.

**SLURM has no internet access on compute nodes.** `load_simclr_backbone(pretrained=True)`
therefore never attempts a network call — it only reads a local file and fails loudly,
with actionable instructions, if that file is not already present. Fetching the
checkpoint is a provisioning step (`prefetch_checkpoint()`), meant to run once from the
login node before a job is submitted, exactly like FreeSurfer/TurboPrep/registration-tool
availability is a provisioning question for the rest of this project rather than
something a compute job resolves for itself.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import resnet18

CHECKPOINT_URL = (
    "https://github.com/emilykaczmarek/3D-Neuro-SimCLR/releases/download/"
    "v1.0.0/simclr_3d_brain_foundation.tar"
)
CHECKPOINT_SHA256 = "f046dafe009de0cfa113a022c1ef9603d123834f217074eb86d26037e72b98f5"
N_FEATURES = 512

# Overridable via SIMCLR_CHECKPOINT_PATH for clusters where $HOME is not shared between
# login and compute nodes (this project's own working tree, under /mnt/lustre/, is
# confirmed shared — the torch hub default under $HOME may or may not be, depending on
# cluster configuration not verifiable from here).
DEFAULT_CHECKPOINT_PATH = Path(
    os.environ.get(
        "SIMCLR_CHECKPOINT_PATH",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "simclr_3d_brain_foundation.tar",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_encoder() -> nn.Module:
    """Construct the bare 3D ResNet-18 encoder, fc stripped to Identity.

    `feed_forward=False` avoids MONAI's `nn.Linear(512, num_classes)` head entirely
    (the upstream `get_resnet(..., num_classes=0)` call would otherwise construct a
    degenerate `nn.Linear(512, 0)` and immediately overwrite it with Identity() in the
    SimCLR wrapper — skipping straight to `feed_forward=False` reaches the same result
    without ever building the degenerate layer).
    """
    return resnet18(spatial_dims=3, n_input_channels=1, num_classes=0, feed_forward=False)


def prefetch_checkpoint(dest: Path | None = None, *, force: bool = False) -> Path:
    """Download and checksum-verify the checkpoint. Run once, from a node with internet.

    Not called by `load_simclr_backbone` — this is a provisioning step, run interactively
    from the login node (or anywhere with outbound internet) before submitting a SLURM
    job that will call `load_simclr_backbone(pretrained=True)`.

    Sets `SSL_CERT_FILE` to `certifi`'s bundle for the duration of this call if it isn't
    already set: this environment's Python `ssl` module defaults to a CA path
    (`/etc/ssl/certs/ca-certificates.crt`, Debian-style) that doesn't exist on this
    RHEL-style system, even though outbound internet and a valid `certifi` bundle are
    both available — confirmed by direct diagnosis. This still performs full certificate
    verification, against a real bundle, not a bypass.
    """
    dest = dest or DEFAULT_CHECKPOINT_PATH
    if dest.exists() and not force:
        if _sha256(dest) == CHECKPOINT_SHA256:
            return dest
        raise ValueError(
            f"{dest} exists but does not match the known checkpoint hash "
            f"({CHECKPOINT_SHA256}). Re-run with force=True to re-download."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    ssl_cert_file_was_unset = "SSL_CERT_FILE" not in os.environ
    if ssl_cert_file_was_unset:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    try:
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(dest))
    finally:
        if ssl_cert_file_was_unset:
            os.environ.pop("SSL_CERT_FILE", None)

    actual_hash = _sha256(dest)
    if actual_hash != CHECKPOINT_SHA256:
        dest.unlink()
        raise ValueError(
            f"Downloaded checkpoint hash {actual_hash} does not match expected "
            f"{CHECKPOINT_SHA256} — deleted the corrupted download. Re-run prefetch_checkpoint()."
        )
    return dest


def load_simclr_backbone(
    *, pretrained: bool, device: str = "cpu", checkpoint_path: Path | None = None
) -> nn.Module:
    """Load the SimCLR encoder — trained checkpoint or PyTorch-default random init.

    pretrained=True: reads a **local file only** — never attempts a network download.
    Raises `FileNotFoundError` with explicit instructions if the checkpoint isn't already
    present at `checkpoint_path` (default: `DEFAULT_CHECKPOINT_PATH`), and `ValueError` if
    it's present but doesn't match the known-good sha256 (protects against a corrupted or
    substituted file, since a SLURM job cannot verify provenance for itself once decoupled
    from the direct download).

    Uses `weights_only=True` when loading — this checkpoint is a plain dict of
    tensors/ints, so full unpickling is not needed, and restricting to the safe
    tensor-loading path avoids executing arbitrary pickled objects from disk.

    pretrained=False: PyTorch's default init (Kaiming/uniform per-layer defaults,
    `BatchNorm3d` running stats at their untrained defaults of mean 0 / var 1) — this is
    exactly the object `gate0.run_gate0` exists to validate, not a stand-in for it. No
    file access at all in this branch.
    """
    encoder = _build_encoder()

    if not pretrained:
        return encoder.to(device).eval()

    path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"SimCLR checkpoint not found at {path}. This function does not download "
            "automatically — SLURM compute nodes have no internet access. Run "
            "prefetch_checkpoint() once from the login node (or any node with internet) "
            "before submitting a job that calls load_simclr_backbone(pretrained=True)."
        )
    actual_hash = _sha256(path)
    if actual_hash != CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint at {path} has hash {actual_hash}, expected {CHECKPOINT_SHA256}. "
            "Re-run prefetch_checkpoint(force=True) rather than trusting this file."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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
