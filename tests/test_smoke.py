"""Synthetic smoke test for PixPoint3D.

This script doesn't need the MVTec3D-AD dataset. It generates a synthetic
"good" sample and verifies that every component of the pipeline executes
without errors:

    1. Multi-view rendering with **fixed** rotation matrices
    2. RGB feature extraction (ImageNet-pretrained backbone runs on CPU)
    3. PointNet++ forward (trainable, since no pre-trained weights here)
    4. Inter-modality MLPs and cosine alignment loss back-prop
    5. Multivariate Gaussian bank update / Mahalanobis distance
    6. Dense (H, W) anomaly map via KNN inverse-distance interpolation
    7. Re-running forward with the same rotations is deterministic
    8. `trainable_parameters()` includes PointNet++ when it is *not* frozen

Run from the project root with::

    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import PixPoint3D  # noqa: E402
from utils import MultivariateGaussianBank, densify_score_map  # noqa: E402
from utils.multiview import random_rotation_matrices  # noqa: E402


def make_synthetic_sample(num_points: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=num_points)
    phi = rng.uniform(0, 2 * np.pi, size=num_points)
    R, r = 0.6, 0.2
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    points -= points.mean(0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    points /= max(scale, 1e-6)
    colors = (rng.uniform(0.3, 1.0, size=(num_points, 3))).astype(np.float32)
    return torch.from_numpy(points), torch.from_numpy(colors)


def main() -> None:
    torch.manual_seed(0)
    num_points = 512
    img_size = 64
    num_views = 3

    model = PixPoint3D(
        d_2d=512,
        d_3d=128,
        img_size=img_size,
        num_views=num_views,
        backbone_2d="resnet18",
        freeze_pointnet=False,
    )
    trainable = model.trainable_parameters()
    n_train = sum(p.numel() for p in trainable)
    print(f"[smoke] #trainable params (MLPs + PointNet++): {n_train:,}")
    pn_count = sum(p.numel() for p in model.point_extractor.parameters())
    fu_count = sum(p.numel() for p in model.fusion.parameters())
    assert n_train == pn_count + fu_count, "trainable params must equal MLPs + PointNet++"

    fixed_rotations = random_rotation_matrices(
        num_views=num_views,
        rot_x_range=(-90.0, 90.0),
        rot_y_range=(-10.0, 10.0),
        rot_z_range=(-10.0, 10.0),
        include_identity=True,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(7),
    )
    print(f"[smoke] fixed_rotations: shape={tuple(fixed_rotations.shape)}")

    points, colors = make_synthetic_sample(num_points)
    torch.manual_seed(123)
    out_a = model(points, colors, rotations=fixed_rotations)
    torch.manual_seed(123)
    out_b = model(points, colors, rotations=fixed_rotations)
    determ_2d = (out_a.f_2d - out_b.f_2d).abs().max().item()
    determ_full = (out_a.fused - out_b.fused).abs().max().item()
    print(f"[smoke] same-rotation determinism: |Δf_2d|_max = {determ_2d:.2e}, "
          f"|Δfused|_max = {determ_full:.2e}")
    assert determ_2d < 1e-5, "2D un-projected features must be deterministic with same rotations"
    assert determ_full < 1e-4, "fused features should match when PointNet++ RNG is seeded"

    out = out_a
    print("[smoke] fused        :", tuple(out.fused.shape))
    print("[smoke] f_2d         :", tuple(out.f_2d.shape))
    print("[smoke] f_3d         :", tuple(out.f_3d.shape))
    print("[smoke] f_hat_2d     :", tuple(out.f_hat_2d.shape))
    print("[smoke] f_hat_3d     :", tuple(out.f_hat_3d.shape))
    print("[smoke] valid_mask   :", tuple(out.valid_mask.shape))
    print("[smoke] point->pixel :", tuple(out.point_to_pixel.shape))

    loss = model.fusion.cosine_alignment_loss(
        out.f_2d, out.f_hat_2d, out.f_3d, out.f_hat_3d
    )
    optim = torch.optim.Adam(trainable, lr=5e-3)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"[smoke] cosine loss   : {loss.item():.4f}")

    bank = MultivariateGaussianBank(num_points=num_points, feat_dim=model.fused_dim)
    model.eval()
    for s in range(3):
        points_i, colors_i = make_synthetic_sample(num_points, seed=s)
        with torch.no_grad():
            out_i = model(points_i, colors_i, rotations=fixed_rotations)
        bank.update(out_i.fused.detach().unsqueeze(0))
    bank.finalise(alpha=0.09)

    points_t, colors_t = make_synthetic_sample(num_points, seed=999)
    with torch.no_grad():
        out_t = model(points_t, colors_t, rotations=fixed_rotations)
        scores = bank.mahalanobis(out_t.fused.detach())
    print(f"[smoke] Mahalanobis  : shape={tuple(scores.shape)}, "
          f"min={float(scores.min()):.3f}, max={float(scores.max()):.3f}")

    anchors = np.random.default_rng(1).integers(0, img_size, size=(num_points, 2))
    dense = densify_score_map(
        scores.cpu().numpy(),
        anchors,
        img_size=img_size,
        k=4,
        smooth_sigma=2.0,
    )
    print(f"[smoke] dense map    : shape={dense.shape}, "
          f"min={float(dense.min()):.3f}, max={float(dense.max()):.3f}, "
          f"non-zero ratio = {float((dense > 1e-6).mean()):.3f}")
    assert dense.shape == (img_size, img_size)
    assert (dense > 1e-6).mean() > 0.5, "dense map should be mostly populated"

    print("[smoke] OK - all components run end-to-end.")


if __name__ == "__main__":
    main()

