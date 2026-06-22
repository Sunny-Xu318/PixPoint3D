"""Multi-view RGB image generation by rotating point clouds.

This module implements Sec. III.C/III.D of the paper:

1. Sample ``M`` rotation matrices in the small angle ranges
   ``theta_x in [-90°, 90°]`` and ``theta_y, theta_z in [-10°, 10°]``.
2. Apply each rotation to the input organised point cloud, then re-project
   the rotated 3D points to 2D using the camera intrinsics provided with the
   dataset (or simple orthographic projection when intrinsics are missing).
3. Splat the per-point RGB colour onto the new 2D canvas, producing M synthetic
   views together with a ``(H, W)`` mapping ``pixel -> point index`` for later
   pixel-to-point feature un-projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch


def _rotation_matrix(theta_x: float, theta_y: float, theta_z: float, device, dtype) -> torch.Tensor:
    """Compose ``R = Rz @ Ry @ Rx`` following Eqs. (1)-(2) of the paper."""
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    cy, sy = math.cos(theta_y), math.sin(theta_y)
    cz, sz = math.cos(theta_z), math.sin(theta_z)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=device, dtype=dtype)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device, dtype=dtype)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=device, dtype=dtype)
    return Rz @ Ry @ Rx


def random_rotation_matrices(
    num_views: int,
    rot_x_range: Tuple[float, float] = (-90.0, 90.0),
    rot_y_range: Tuple[float, float] = (-10.0, 10.0),
    rot_z_range: Tuple[float, float] = (-10.0, 10.0),
    include_identity: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``num_views`` rotation matrices.

    Returns:
        Rs: (num_views, 3, 3) tensor.
    """
    Rs = []
    if include_identity and num_views >= 1:
        Rs.append(torch.eye(3, device=device, dtype=dtype))
    while len(Rs) < num_views:
        if generator is None:
            tx = torch.empty(1).uniform_(*rot_x_range).item()
            ty = torch.empty(1).uniform_(*rot_y_range).item()
            tz = torch.empty(1).uniform_(*rot_z_range).item()
        else:
            tx = torch.empty(1).uniform_(*rot_x_range, generator=generator).item()
            ty = torch.empty(1).uniform_(*rot_y_range, generator=generator).item()
            tz = torch.empty(1).uniform_(*rot_z_range, generator=generator).item()
        Rs.append(
            _rotation_matrix(
                math.radians(tx), math.radians(ty), math.radians(tz), device, dtype
            )
        )
    return torch.stack(Rs[:num_views], dim=0)


@dataclass
class MultiViewRendering:
    """Per-view rendering output.

    Attributes
    ----------
    images : Tensor
        (M, 3, H, W) rendered RGB images in [0, 1].
    point_to_pixel : Tensor
        (M, N, 2) integer pixel coordinates (u, v) for each point in each view,
        clamped to the image size. Values outside the canvas are clamped, so
        callers should also use the ``valid_mask`` below to know if a point
        actually projected inside.
    valid_mask : Tensor
        (M, N) bool mask: True when the point is visible inside the canvas.
    """

    images: torch.Tensor
    point_to_pixel: torch.Tensor
    valid_mask: torch.Tensor


def _orthographic_project(points: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Project (B, N, 3) points to (B, N, 2) pixel coords using a normalised
    orthographic projection of the (x, y) plane to [0, W-1] x [0, H-1].

    This serves as a fallback when no camera intrinsics are provided.
    """
    B, N, _ = points.shape
    xy = points[..., :2]
    mins = xy.amin(dim=1, keepdim=True)
    maxs = xy.amax(dim=1, keepdim=True)
    ranges = (maxs - mins).clamp(min=1e-6)
    norm = (xy - mins) / ranges
    u = norm[..., 0] * (W - 1)
    v = (1.0 - norm[..., 1]) * (H - 1)
    return torch.stack([u, v], dim=-1)


def _perspective_project(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pinhole projection. ``points``: (B, N, 3), ``K``: (3, 3) or (B, 3, 3)."""
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(points.shape[0], -1, -1)
    proj = torch.einsum("bij,bnj->bni", K, points)
    z = proj[..., 2:3].clamp(min=1e-6)
    return proj[..., :2] / z


def generate_multi_view_renderings(
    points: torch.Tensor,
    colors: torch.Tensor,
    img_size: int,
    num_views: int,
    rot_x_range: Tuple[float, float] = (-90.0, 90.0),
    rot_y_range: Tuple[float, float] = (-10.0, 10.0),
    rot_z_range: Tuple[float, float] = (-10.0, 10.0),
    intrinsics: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    rotations: Optional[torch.Tensor] = None,
) -> MultiViewRendering:
    """Render ``num_views`` colorised images by rotating the point cloud.

    Parameters
    ----------
    points : Tensor
        (N, 3) point cloud coordinates.
    colors : Tensor
        (N, 3) RGB color associated with each point, in [0, 1].
    img_size : int
        Output image height/width (square).
    num_views : int
        Number of views to generate (the first view is always the identity
        rotation, which corresponds to the original camera frame).
    intrinsics : Tensor or None
        (3, 3) camera intrinsic matrix. If ``None`` an orthographic fallback
        is used.
    rotations : Tensor or None
        Optional pre-computed ``(num_views, 3, 3)`` rotation matrices. When
        provided, the random sampling step is skipped — this is the recommended
        path at *inference* time to keep the view set identical to the one
        used during Gaussian-bank fitting (Sec. III.I).

    Returns
    -------
    MultiViewRendering
    """
    assert points.dim() == 2 and points.shape[1] == 3
    assert colors.shape == points.shape
    device, dtype = points.device, points.dtype

    if rotations is not None:
        if rotations.shape != (num_views, 3, 3):
            raise ValueError(
                f"`rotations` must be ({num_views}, 3, 3), got {tuple(rotations.shape)}"
            )
        Rs = rotations.to(device=device, dtype=dtype)
    else:
        Rs = random_rotation_matrices(
            num_views,
            rot_x_range=rot_x_range,
            rot_y_range=rot_y_range,
            rot_z_range=rot_z_range,
            include_identity=True,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    rotated = torch.einsum("mij,nj->mni", Rs, points)
    if intrinsics is not None:
        pixels = _perspective_project(rotated, intrinsics.to(device=device, dtype=dtype))
    else:
        pixels = _orthographic_project(rotated, img_size, img_size)

    u = pixels[..., 0].round().long()
    v = pixels[..., 1].round().long()
    valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
    u_c = u.clamp(0, img_size - 1)
    v_c = v.clamp(0, img_size - 1)

    images = torch.zeros(num_views, 3, img_size, img_size, device=device, dtype=dtype)
    z = rotated[..., 2]
    depth_buffer = torch.full(
        (num_views, img_size, img_size), float("inf"), device=device, dtype=dtype
    )
    for m in range(num_views):
        order = torch.argsort(z[m], descending=False)
        u_o = u_c[m, order]
        v_o = v_c[m, order]
        z_o = z[m, order]
        c_o = colors[order]
        flat_idx = v_o * img_size + u_o
        current_depth = depth_buffer[m].view(-1)
        existing = current_depth[flat_idx]
        replace_mask = z_o < existing
        flat_idx_r = flat_idx[replace_mask]
        if flat_idx_r.numel() == 0:
            continue
        current_depth[flat_idx_r] = z_o[replace_mask]
        depth_buffer[m] = current_depth.view(img_size, img_size)
        for ch in range(3):
            channel = images[m, ch].view(-1)
            channel[flat_idx_r] = c_o[replace_mask, ch]
            images[m, ch] = channel.view(img_size, img_size)

    return MultiViewRendering(
        images=images,
        point_to_pixel=torch.stack([u_c, v_c], dim=-1),
        valid_mask=valid,
    )
