"""Utilities to un-project 2D feature maps back onto a 3D point cloud.

The 2M3DF paper (Sec. III.F.1) takes the multi-view RGB feature maps
``F^{RGB}_{vi} in R^{H x W x d_2D}`` and assigns each point the feature of the
pixel it projects onto. Aggregation across the ``M`` views is then a simple
point-wise mean.
"""
from __future__ import annotations

from typing import Optional

import torch


def unproject_features_to_points(
    feature_maps: torch.Tensor,
    point_to_pixel: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample per-view feature maps at projected pixel locations.

    Parameters
    ----------
    feature_maps : Tensor
        ``(M, C, H, W)`` per-view feature maps (already upsampled to the input
        resolution).
    point_to_pixel : Tensor
        ``(M, N, 2)`` integer pixel coordinates ``(u, v)``, expected to be in
        the range ``[0, H-1] x [0, W-1]``.
    valid_mask : Tensor or None
        ``(M, N)`` bool tensor marking points that actually projected inside
        the canvas. Points outside are zeroed and **excluded** from the mean.

    Returns
    -------
    Tensor
        ``(N, C)`` aggregated point-wise feature obtained by per-view sampling
        and average pooling (Eq. (6) of the paper).
    """
    M, C, H, W = feature_maps.shape
    _, N, _ = point_to_pixel.shape
    device = feature_maps.device

    u = point_to_pixel[..., 0].long().clamp(0, W - 1)
    v = point_to_pixel[..., 1].long().clamp(0, H - 1)

    feat_per_view = []
    for m in range(M):
        fm = feature_maps[m].permute(1, 2, 0)
        sampled = fm[v[m], u[m]]
        feat_per_view.append(sampled)
    feat_per_view = torch.stack(feat_per_view, dim=0)

    if valid_mask is not None:
        weight = valid_mask.to(feat_per_view.dtype).unsqueeze(-1)
        weighted = feat_per_view * weight
        denom = weight.sum(dim=0).clamp(min=1e-6)
        fused = weighted.sum(dim=0) / denom
    else:
        fused = feat_per_view.mean(dim=0)
    return fused
