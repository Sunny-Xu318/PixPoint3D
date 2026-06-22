"""Evaluation entry-point for PixPoint3D.

Loads the trained cross-modal MLPs, the multivariate Gaussian bank, the
shared anchor positions and the *fixed* rotation matrices used to fit the bank.
It then runs inference over the *test* split of one MVTec3D-AD category and
reports the I-AUROC and AUPRO metrics (matching Tables I & II of the paper).

The per-pixel anomaly score map is reconstructed from the ``N`` per-anchor
Mahalanobis distances via KNN inverse-distance weighting plus an optional
Gaussian smoothing (:func:`utils.densify.densify_score_map`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import MVTec3DCategoryDataset  # noqa: E402
from models import PixPoint3D  # noqa: E402
from utils import MultivariateGaussianBank, densify_score_map  # noqa: E402
from utils.metrics import compute_aupro, compute_image_auroc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PixPoint3D on one MVTec3D-AD category.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_views", type=int, default=7)
    p.add_argument("--d_2d", type=int, default=512)
    p.add_argument("--d_3d", type=int, default=128)
    p.add_argument("--backbone_2d", type=str, default="wide_resnet50_2")
    p.add_argument("--ransac_distance", type=float, default=0.005)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--knn_k", type=int, default=4,
                   help="K for KNN densification (default 4).")
    p.add_argument("--smooth_sigma", type=float, default=4.0,
                   help="Gaussian smoothing sigma on the dense map (0 = off).")
    p.add_argument("--knn_power", type=float, default=2.0,
                   help="Inverse-distance exponent used by KNN densification.")
    return p.parse_args()


def collate_single(batch):
    assert len(batch) == 1
    return batch[0]


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)

    anchors = np.load(ckpt_dir / "anchors.npy")
    print(f"[PixPoint3D] anchors loaded: shape={anchors.shape}")

    rotations_path = ckpt_dir / "rotations.pt"
    if not rotations_path.exists():
        raise FileNotFoundError(
            f"`rotations.pt` not found under {ckpt_dir}. "
            "Re-train with the latest train.py so that the fixed view set is saved."
        )
    fixed_rotations = torch.load(rotations_path, map_location=args.device).to(args.device)
    if fixed_rotations.shape[0] != args.num_views:
        print(
            f"[PixPoint3D] WARNING: --num_views={args.num_views} but checkpoint stored "
            f"{fixed_rotations.shape[0]} views. Using {fixed_rotations.shape[0]}."
        )
        args.num_views = fixed_rotations.shape[0]

    config_path = ckpt_dir / "config.pt"
    freeze_pn_in_ckpt = True
    if config_path.exists():
        cfg = torch.load(config_path, map_location="cpu")
        freeze_pn_in_ckpt = bool(cfg.get("freeze_pointnet", True))

    num_points = int(anchors.shape[0])
    test_set = MVTec3DCategoryDataset(
        root=args.data_root,
        category=args.category,
        split="test",
        img_size=args.img_size,
        num_points=num_points,
        anchors=anchors,
        ransac_distance=args.ransac_distance,
        seed=0,
    )
    print(f"[PixPoint3D] #test samples = {len(test_set)}")
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_single
    )

    model = PixPoint3D(
        d_2d=args.d_2d,
        d_3d=args.d_3d,
        img_size=args.img_size,
        num_views=args.num_views,
        backbone_2d=args.backbone_2d,
        freeze_pointnet=freeze_pn_in_ckpt,
    ).to(args.device)
    model.eval()
    model.fusion.phi_2d_to_3d.load_state_dict(
        torch.load(ckpt_dir / "phi_2d_to_3d.pth", map_location=args.device)
    )
    model.fusion.phi_3d_to_2d.load_state_dict(
        torch.load(ckpt_dir / "phi_3d_to_2d.pth", map_location=args.device)
    )
    pn_path = ckpt_dir / "pointnet.pth"
    if pn_path.exists():
        model.point_extractor.net.load_state_dict(
            torch.load(pn_path, map_location=args.device)
        )
        print(f"[PixPoint3D] loaded fine-tuned PointNet++ from {pn_path}")

    bank = MultivariateGaussianBank.load(ckpt_dir / "gaussian.pt", device=args.device)

    image_scores: List[float] = []
    image_labels: List[int] = []
    score_maps: List[np.ndarray] = []
    gt_masks: List[np.ndarray] = []

    with torch.no_grad():
        for sample in tqdm(test_loader, desc=f"test/{args.category}"):
            points = sample["points"].to(args.device)
            colors = sample["colors"].to(args.device)
            out = model(points, colors, rotations=fixed_rotations)
            point_scores = bank.mahalanobis(out.fused)

            image_scores.append(float(point_scores.max().item()))
            image_labels.append(int(sample["label"].item()))

            dense_map = densify_score_map(
                point_scores.detach().cpu().numpy(),
                anchors,
                img_size=args.img_size,
                k=args.knn_k,
                smooth_sigma=args.smooth_sigma,
                power=args.knn_power,
            )
            score_maps.append(dense_map)
            gt_masks.append(sample["mask"].numpy())

    img_auroc = compute_image_auroc(image_scores, image_labels)
    aupro = compute_aupro(gt_masks, score_maps)
    print(f"[PixPoint3D] {args.category:>12s} | I-AUROC = {img_auroc:.4f} | AUPRO = {aupro:.4f}")


if __name__ == "__main__":
    main()

