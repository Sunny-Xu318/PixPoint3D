"""Frozen feature extractors used by 2M3DF.

- ``RGBFeatureExtractor``: takes a batch of multi-view RGB images and returns
  the spatial feature map produced by the 3rd block of a WideResNet-50
  pre-trained on ImageNet. The map is bilinearly upsampled back to the input
  resolution so that the channel dim acts as a per-pixel descriptor of size
  ``d_2D = 512`` (the paper's choice).
- ``PointFeatureExtractor``: thin wrapper around :class:`PointNet2Seg` producing
  per-point features of size ``d_3D = 128``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .pointnet2 import PointNet2Seg


_BACKBONES = {
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
    "resnet50":        (models.resnet50,        models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet18":        (models.resnet18,        models.ResNet18_Weights.IMAGENET1K_V1),
}


class RGBFeatureExtractor(nn.Module):
    """Frozen 2D feature extractor based on layer3 of a (Wide-)ResNet.

    Parameters
    ----------
    backbone : str
        One of ``{"wide_resnet50_2", "resnet50", "resnet18"}``.
    layer : str
        Which residual block's output to use as feature map. Defaults to
        ``"layer3"`` which gives ``d_2D = 512`` for (Wide-)ResNet-50.
    out_size : int
        Spatial size of the returned feature map. The feature map is bilinearly
        upsampled to this size so the channels become per-pixel descriptors.
    """

    LAYER_DIM = {
        "wide_resnet50_2": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
        "resnet50":        {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
        "resnet18":        {"layer1": 64,  "layer2": 128, "layer3": 256,  "layer4": 512},
    }

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layer: str = "layer3",
        out_size: int = 224,
        project_dim: int = 512,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}")
        ctor, weights = _BACKBONES[backbone]
        net = ctor(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.layer = layer
        self.out_size = out_size

        raw_dim = self.LAYER_DIM[backbone][layer]
        if raw_dim != project_dim:
            self.proj = nn.Conv2d(raw_dim, project_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.out_dim = project_dim

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):  # noqa: D401
        """Keep the backbone in eval mode regardless of caller (BN frozen)."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) - RGB images in [0,1] range, already normalized with
            ImageNet stats outside this module.
        Returns:
            feat: (B, d_2D, H, W) feature map bilinearly upsampled to (H, W).
        """
        h = self.stem(x)
        h = self.layer1(h)
        if self.layer == "layer1":
            f = h
        else:
            h = self.layer2(h)
            if self.layer == "layer2":
                f = h
            else:
                h = self.layer3(h)
                if self.layer == "layer3":
                    f = h
                else:
                    h = self.layer4(h)
                    f = h
        f = self.proj(f)
        f = F.interpolate(f, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return f


class PointFeatureExtractor(nn.Module):
    """Wrapper around :class:`PointNet2Seg` that keeps the network frozen.

    Parameters
    ----------
    out_dim : int
        Per-point feature dimension. Defaults to 128 (paper's d_3D).
    pretrained_ckpt : str or None
        Optional path to a pre-trained PointNet++ checkpoint. If provided, the
        weights are loaded with ``strict=False`` so partial checkpoints are
        accepted.
    freeze : bool
        Whether to freeze the network (default True, matching the paper).
    """

    def __init__(
        self,
        out_dim: int = 128,
        pretrained_ckpt: str | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.net = PointNet2Seg(out_dim=out_dim)
        self.out_dim = out_dim
        self._frozen = bool(freeze)
        if pretrained_ckpt is not None:
            state = torch.load(pretrained_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.net.load_state_dict(state, strict=False)
        if self._frozen:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def train(self, mode: bool = True):  # noqa: D401
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, N, 3)
        Returns: (B, N, d_3D)
        """
        x = xyz.transpose(1, 2).contiguous()
        if self._frozen:
            with torch.no_grad():
                return self.net(x)
        return self.net(x)
