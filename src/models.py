from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSepConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class CNN3D_DoubleConv(nn.Module):
    """Frozen untrained 3D CNN returning 7680-dimensional multiscale features."""

    def __init__(self, in_channels=3, use_norm=True, activation="relu"):
        super().__init__()
        self.out_features = 8 * (64 + 128 + 256 + 512)
        self.act = _activation(activation)

        self.conv1a = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.norm1a = nn.GroupNorm(8, 64) if use_norm else nn.Identity()
        self.conv1b = nn.Conv3d(64, 64, 3, padding=1)
        self.norm1b = nn.GroupNorm(8, 64) if use_norm else nn.Identity()
        self.down1 = nn.AvgPool3d(2)

        self.conv2a = DepthwiseSepConv3d(64, 128)
        self.norm2a = nn.GroupNorm(8, 128) if use_norm else nn.Identity()
        self.conv2b = DepthwiseSepConv3d(128, 128)
        self.norm2b = nn.GroupNorm(8, 128) if use_norm else nn.Identity()
        self.down2 = nn.AvgPool3d(2)

        self.conv3a = DepthwiseSepConv3d(128, 256)
        self.norm3a = nn.GroupNorm(8, 256) if use_norm else nn.Identity()
        self.conv3b = DepthwiseSepConv3d(256, 256)
        self.norm3b = nn.GroupNorm(8, 256) if use_norm else nn.Identity()
        self.down3 = nn.AvgPool3d(2)

        self.conv4a = DepthwiseSepConv3d(256, 512)
        self.norm4a = nn.GroupNorm(8, 512) if use_norm else nn.Identity()
        self.conv4b = DepthwiseSepConv3d(512, 512)
        self.norm4b = nn.GroupNorm(8, 512) if use_norm else nn.Identity()
        self.down4 = nn.AvgPool3d(2)

        self.pool = nn.AdaptiveAvgPool3d(2)

    def forward(self, x):
        x1 = self.act(self.norm1a(self.conv1a(x)))
        x1 = self.down1(self.act(self.norm1b(self.conv1b(x1))))
        x2 = self.act(self.norm2a(self.conv2a(x1)))
        x2 = self.down2(self.act(self.norm2b(self.conv2b(x2))))
        x3 = self.act(self.norm3a(self.conv3a(x2)))
        x3 = self.down3(self.act(self.norm3b(self.conv3b(x3))))
        x4 = self.act(self.norm4a(self.conv4a(x3)))
        x4 = self.down4(self.act(self.norm4b(self.conv4b(x4))))
        return torch.cat(
            [
                torch.flatten(self.pool(x1), 1),
                torch.flatten(self.pool(x2), 1),
                torch.flatten(self.pool(x3), 1),
                torch.flatten(self.pool(x4), 1),
            ],
            dim=1,
        )


class CNN3D_CovPool(nn.Module):
    """Frozen untrained 3D CNN returning covariance-pooled multiscale features.

    Feature dim per scale = (C * 8) + max_ch*(max_ch-1)//2  where max_ch=32.
    Total with default channels: 1008 + 1520 + 2544 + 4592 = 9664.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.out_features = (
            (64 * 8 + 32 * 31 // 2)
            + (128 * 8 + 32 * 31 // 2)
            + (256 * 8 + 32 * 31 // 2)
            + (512 * 8 + 32 * 31 // 2)
        )

        self.conv1a = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.norm1a = nn.GroupNorm(8, 64)
        self.conv1b = nn.Conv3d(64, 64, 3, padding=1)
        self.norm1b = nn.GroupNorm(8, 64)
        self.down1 = nn.AvgPool3d(2)

        self.conv2a = DepthwiseSepConv3d(64, 128)
        self.norm2a = nn.GroupNorm(8, 128)
        self.conv2b = DepthwiseSepConv3d(128, 128)
        self.norm2b = nn.GroupNorm(8, 128)
        self.down2 = nn.AvgPool3d(2)

        self.conv3a = DepthwiseSepConv3d(128, 256)
        self.norm3a = nn.GroupNorm(8, 256)
        self.conv3b = DepthwiseSepConv3d(256, 256)
        self.norm3b = nn.GroupNorm(8, 256)
        self.down3 = nn.AvgPool3d(2)

        self.conv4a = DepthwiseSepConv3d(256, 512)
        self.norm4a = nn.GroupNorm(8, 512)
        self.conv4b = DepthwiseSepConv3d(512, 512)
        self.norm4b = nn.GroupNorm(8, 512)
        self.down4 = nn.AvgPool3d(2)

        self.avg_pool = nn.AdaptiveAvgPool3d(2)

    def _cov_pool(self, x, max_ch=32):
        b, c = x.size(0), min(x.size(1), max_ch)
        mean_feats = self.avg_pool(x).view(b, -1)
        xc = x[:, :c]
        flat = xc.view(b, c, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / (flat.size(2) - 1)
        idx = torch.triu_indices(c, c, offset=1)
        cov_feats = cov[:, idx[0], idx[1]]
        return torch.cat([mean_feats, cov_feats], dim=1)

    def forward(self, x):
        x1 = F.relu(self.norm1a(self.conv1a(x)))
        x1 = self.down1(F.relu(self.norm1b(self.conv1b(x1))))
        x2 = F.relu(self.norm2a(self.conv2a(x1)))
        x2 = self.down2(F.relu(self.norm2b(self.conv2b(x2))))
        x3 = F.relu(self.norm3a(self.conv3a(x2)))
        x3 = self.down3(F.relu(self.norm3b(self.conv3b(x3))))
        x4 = F.relu(self.norm4a(self.conv4a(x3)))
        x4 = self.down4(F.relu(self.norm4b(self.conv4b(x4))))
        return torch.cat([
            self._cov_pool(x1, 32),
            self._cov_pool(x2, 32),
            self._cov_pool(x3, 32),
            self._cov_pool(x4, 32),
        ], dim=1)


def init_xavier(model):
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
