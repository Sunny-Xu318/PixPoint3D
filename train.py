"""Training entry-point for PixPoint3D.

The pipeline is split into **three** phases:

* **Phase 0 �?Shared anchor computation.** A small subset of the training set
  is used to build a per-pixel foreground vote map; the resulting reliable
  foreground region is sampled to produce ``num_points`` deterministic
  ``(u, v)`` pixel positions shared by every sample. This is what makes the
  per-point Gaussian bank (Eq. 12�?3) actually meaningful across samples.
* **Phase 1 �?Cross-modal MLP fitting.** The 2D backbone is always frozen.
  PointNet++ is either frozen (when ``--pretrained_pointnet`` is given) or
  jointly fine-tuned. We optimise ``phi_2d_to_3d`` and ``phi_3d_to_2d`` (plus
  optionally PointNet++) with Adam (lr 5e-3) and the cosine alignment loss
  (Eq. 9). Random per-iteration rotations act as data augmentation.
* **Phase 2 �?Multivariate Gaussian fitting.** With the trained MLPs we run
  one extra pass over the training split using a *fixed* rotation set (saved
  with the checkpoint) and accumulate fused features per point. The bank is
  finalised with regulariser ``alpha = 0.09``.

All artefacts (MLPs, optional PointNet++, anchors, rotations, Gaussian bank,
config) are saved to ``--save_dir/<category>/`` for ``test.py`` to load.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import MVTec3DCategoryDataset, compute_shared_anchors  # noqa: E402
from models import PixPoint3D  # noqa: E402
from utils import MultivariateGaussianBank  # noqa: E402
from utils.multiview import random_rotation_matrices  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PixPoint3D on one MVTec3D-AD category.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_points", type=int, default=4096)
    p.add_argument("--num_views", type=int, default=7)
    p.add_argument("--d_2d", type=int, default=512)
    p.add_argument("--d_3d", type=int, default=128)
    p.add_argument("--backbone_2d", type=str, default="wide_resnet50_2")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--alpha", type=float, default=0.09)
    p.add_argument("--ransac_distance", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrained_pointnet", type=str, default=None,
                   help="Optional path to a pre-trained PointNet++ checkpoint. When "
                        "omitted, PointNet++ is trained jointly with the fusion MLPs "
                        "(deviates from the paper's frozen-backbone setting, but is "
                        "necessary when no pre-trained 3D weights are available).")
    p.add_argument("--freeze_pointnet", action="store_true",
                   help="Force PointNet++ to stay frozen (paper-faithful). Only "
                        "useful when combined with --pretrained_pointnet.")
    p.add_argument("--fg_vote_threshold", type=float, default=0.5,
                   help="Anchor-vote threshold for compute_shared_anchors.")
    p.add_argument("--max_samples_for_anchor", type=int, default=50)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def collate_single(batch):
    """Custom collate fn �?we keep batch size = 1 because per-sample point
    counts are identical but the renderings are sample-specific."""
    assert len(batch) == 1
    return batch[0]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir) / args.category
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PixPoint3D] category={args.category}  device={args.device}")

    print("[PixPoint3D] Phase 0: computing shared anchor positions...")
    bootstrap_set = MVTec3DCategoryDataset(
        root=args.data_root,
        category=args.category,
        split="train",
        img_size=args.img_size,
        num_points=args.num_points,
        anchors=None,
        ransac_distance=args.ransac_distance,
        seed=args.seed,
    )
    anchors = compute_shared_anchors(
        samples=bootstrap_set.samples,
        num_points=args.num_points,
        img_size=args.img_size,
        ransac_distance=args.ransac_distance,
        seed=args.seed,
        max_samples_for_vote=args.max_samples_for_anchor,
        fg_vote_threshold=args.fg_vote_threshold,
    )
    np.save(save_dir / "anchors.npy", anchors)
    print(f"[PixPoint3D] anchors: shape={anchors.shape}  saved to {save_dir/'anchors.npy'}")

    train_set = MVTec3DCategoryDataset(
        root=args.data_root,
        category=args.category,
        split="train",
        img_size=args.img_size,
        num_points=args.num_points,
        anchors=anchors,
        ransac_distance=args.ransac_distance,
        seed=args.seed,
    )
    print(f"[PixPoint3D] #train samples = {len(train_set)}")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_single,
    )

    freeze_pn = bool(args.freeze_pointnet)
    if args.pretrained_pointnet is None and freeze_pn:
        print("[PixPoint3D] WARNING: --freeze_pointnet given without --pretrained_pointnet; "
              "the 3D features will be random. Pass --pretrained_pointnet to load weights.")
    model = PixPoint3D(
        d_2d=args.d_2d,
        d_3d=args.d_3d,
        img_size=args.img_size,
        num_views=args.num_views,
        backbone_2d=args.backbone_2d,
        pretrained_pointnet=args.pretrained_pointnet,
        freeze_pointnet=freeze_pn,
    ).to(args.device)
    print(f"[PixPoint3D] PointNet++ frozen: {freeze_pn}")
    trainable = model.trainable_parameters()
    print(f"[PixPoint3D] #trainable parameters = {sum(p.numel() for p in trainable):,}")

    optim = torch.optim.Adam(trainable, lr=args.lr)

    print("[PixPoint3D] Phase 1: fitting cross-modal MLPs (random rotations as aug.)...")
    for epoch in range(args.epochs):
        model.fusion.train()
        if not freeze_pn:
            model.point_extractor.train()
        running = 0.0
        for sample in train_loader:
            points = sample["points"].to(args.device)
            colors = sample["colors"].to(args.device)
            out = model(points, colors)
            loss = model.fusion.cosine_alignment_loss(
                out.f_2d, out.f_hat_2d, out.f_3d, out.f_hat_3d
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
        avg = running / max(len(train_loader), 1)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch+1:3d}/{args.epochs}   loss = {avg:.4f}")

    torch.save(model.fusion.phi_2d_to_3d.state_dict(), save_dir / "phi_2d_to_3d.pth")
    torch.save(model.fusion.phi_3d_to_2d.state_dict(), save_dir / "phi_3d_to_2d.pth")
    if not freeze_pn:
        torch.save(model.point_extractor.net.state_dict(), save_dir / "pointnet.pth")
    torch.save({
        "args": vars(args),
        "fused_dim": model.fused_dim,
        "freeze_pointnet": freeze_pn,
    }, save_dir / "config.pt")

    print("[PixPoint3D] Phase 2: fitting multivariate Gaussian bank with fixed rotations...")
    fixed_rotations = random_rotation_matrices(
        num_views=args.num_views,
        rot_x_range=tuple(model.rot_x_range),
        rot_y_range=tuple(model.rot_y_range),
        rot_z_range=tuple(model.rot_z_range),
        include_identity=True,
        device=args.device,
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    torch.save(fixed_rotations.cpu(), save_dir / "rotations.pt")

    model.eval()
    bank = MultivariateGaussianBank(
        num_points=args.num_points,
        feat_dim=model.fused_dim,
        device=args.device,
    )
    with torch.no_grad():
        for sample in tqdm(train_loader, desc="bank"):
            points = sample["points"].to(args.device)
            colors = sample["colors"].to(args.device)
            out = model(points, colors, rotations=fixed_rotations)
            bank.update(out.fused.detach().unsqueeze(0))
    bank.finalise(alpha=args.alpha)
    bank.save(save_dir / "gaussian.pt")

    print(f"[PixPoint3D] Done. Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()

