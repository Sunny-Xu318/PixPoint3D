"""Pure PyTorch implementation of PointNet++ (segmentation) used by 2M3DF.

The implementation follows Qi et al. 2017 (NeurIPS) and does **not** rely on
any custom CUDA op. All ops are written with vanilla PyTorch and therefore work
on both Linux and Windows.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared euclidean distance.

    Args:
        src: (B, N, C)
        dst: (B, M, C)
    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)
    Returns:
        new_points: same leading dims as idx, last dim C
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative farthest point sampling.

    Args:
        xyz: (B, N, 3)
        npoint: number of points to sample
    Returns:
        centroids: (B, npoint) long indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Find at most nsample neighbours inside `radius` for each query.

    Args:
        radius: search radius
        nsample: max neighbours
        xyz: (B, N, 3) candidate points
        new_xyz: (B, S, 3) query points
    Returns:
        group_idx: (B, S, nsample) long indices, padded with the first neighbour
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long, device=device)
        .view(1, 1, N)
        .repeat(B, S, 1)
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0:1].repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FPS + ball-query group used by PointNetSetAbstraction.

    Args:
        npoint: # sampled centroids
        radius: ball-query radius
        nsample: # neighbours per centroid
        xyz: (B, N, 3)
        points: (B, N, D) or None
    Returns:
        new_xyz: (B, npoint, 3)
        new_points: (B, npoint, nsample, 3 + D)
    """
    B, N, _ = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, 3)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor, points: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Special group: single centroid at origin covering all points."""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """Standard Set Abstraction layer used in PointNet++."""

    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp: List[int],
        group_all: bool,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz:    (B, 3, N)
        points: (B, D, N) or None
        Returns:
            new_xyz:    (B, 3, S)
            new_points: (B, mlp[-1], S)
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            assert self.npoint is not None and self.radius is not None and self.nsample is not None
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """Feature propagation (interpolation) layer used in PointNet++ Seg."""

    def __init__(self, in_channel: int, mlp: List[int]) -> None:
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """
        xyz1:    (B, 3, N)   target (fine) points
        xyz2:    (B, 3, S)   source (coarse) points
        points1: (B, D1, N)  optional skip features
        points2: (B, D2, S)  features to be interpolated
        Returns:
            new_points: (B, mlp[-1], N)
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2Seg(nn.Module):
    """PointNet++ segmentation backbone producing per-point features of size ``out_dim``.

    The architecture follows the original PointNet++ semantic segmentation
    head (with 4 SA layers + 4 FP layers), and projects the final per-point
    descriptor to ``out_dim`` channels using a 1×1 conv. Default ``out_dim=128``
    matches the paper's d_3D.
    """

    def __init__(self, out_dim: int = 128, in_channel: int = 0) -> None:
        super().__init__()
        self.in_channel = in_channel
        c_xyz = 3
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            in_channel=c_xyz + in_channel, mlp=[32, 32, 64], group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            in_channel=c_xyz + 64, mlp=[64, 64, 128], group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            in_channel=c_xyz + 128, mlp=[128, 128, 256], group_all=False,
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            in_channel=c_xyz + 256, mlp=[256, 256, 512], group_all=False,
        )

        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_channel, mlp=[128, 128])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_dim, 1),
        )

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        xyz: (B, 3, N)
        features: (B, D, N) or None
        Returns: (B, N, out_dim)
        """
        l0_xyz = xyz
        l0_points = features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        per_point = self.head(l0_points)
        per_point = per_point.permute(0, 2, 1).contiguous()
        return per_point
