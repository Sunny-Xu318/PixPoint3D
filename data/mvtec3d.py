"""MVTec 3D-AD dataset loader.

The dataset is organised as::

    root/<category>/train/good/{rgb,xyz}/000.{png,tiff}
    root/<category>/test/<defect_type>/{rgb,xyz,gt}/000.{png,tiff,png}
    root/<category>/validation/good/...

Where ``rgb`` are H×W×3 colour images, ``xyz`` are H×W×3 organised point clouds
saved as 32-bit floating point TIFFs and ``gt`` (only in the test split) are
H×W binary masks (255 inside the defect).

The dataset has *no* annotated background; we follow the paper and use a RANSAC
plane-fit to remove the dominant background plane.

**Shared-anchor sampling.**
The paper fits a multivariate Gaussian distribution *per point position*,
which implicitly requires that point index ``i`` corresponds to the **same
physical location** across all training/testing samples. To preserve that
invariant we provide :func:`compute_shared_anchors`, which produces ``N``
deterministic pixel positions shared by every sample of one category. Each
sample then extracts ``(xyz, rgb)`` exactly at those pixel positions, so the
per-point statistics fed into the Gaussian bank are comparable across samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import tifffile
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`tifffile` is required to read MVTec3D-AD point clouds. "
        "Install it with `pip install tifffile`."
    ) from e


_DEFAULT_CATEGORIES = (
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
)


@dataclass
class MVTecSample:
    rgb: torch.Tensor                # (3, H, W) in [0, 1]
    points: torch.Tensor             # (N, 3)
    colors: torch.Tensor             # (N, 3)
    pixel_index: torch.Tensor        # (N, 2) the (u, v) the points came from
    label: int                       # 0 (good) / 1 (defect)
    mask: Optional[torch.Tensor]     # (H, W) binary, only set for test split
    category: str
    defect: str
    path: str


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), resample=Image.BICUBIC)
    return np.asarray(pil)


def _resize_xyz(xyz: np.ndarray, size: int) -> np.ndarray:
    """Resize an organised point cloud using nearest-neighbour to avoid
    averaging 3D positions of different surfaces.
    """
    out = np.zeros((size, size, 3), dtype=xyz.dtype)
    for c in range(3):
        pil = Image.fromarray(xyz[..., c])
        pil = pil.resize((size, size), resample=Image.NEAREST)
        out[..., c] = np.asarray(pil)
    return out


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask)
    pil = pil.resize((size, size), resample=Image.NEAREST)
    return np.asarray(pil)


def _ransac_plane(points: np.ndarray, num_iter: int = 200, distance: float = 0.005) -> np.ndarray:
    """Estimate the dominant plane via RANSAC. Returns a (4,) plane (a,b,c,d)
    with unit normal. Operates only on valid (non-zero) points to be robust.
    """
    valid = np.linalg.norm(points, axis=1) > 1e-6
    pts = points[valid]
    if pts.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0, 0.0])
    best_inliers, best_plane = -1, None
    rng = np.random.default_rng(0)
    for _ in range(num_iter):
        idx = rng.choice(pts.shape[0], 3, replace=False)
        p1, p2, p3 = pts[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-6:
            continue
        n = n / np.linalg.norm(n)
        d = -np.dot(n, p1)
        dist = np.abs(pts @ n + d)
        inliers = int((dist < distance).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = np.concatenate([n, [d]])
    return best_plane if best_plane is not None else np.array([0.0, 0.0, 1.0, 0.0])


def _remove_background(
    points: np.ndarray, rgb: np.ndarray, distance: float = 0.005
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero out background points (closer to the dominant plane than ``distance``).

    Parameters
    ----------
    points : (H, W, 3) organised point cloud.
    rgb    : (H, W, 3) colour image associated with the points.

    Returns
    -------
    points_out, rgb_out, fg_mask  (all H × W).
    """
    H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    plane = _ransac_plane(pts, distance=distance)
    n, d = plane[:3], plane[3]
    signed_dist = pts @ n + d
    bg = (np.abs(signed_dist) < distance) | (np.linalg.norm(pts, axis=1) < 1e-6)
    fg = ~bg
    points_out = points.copy().reshape(-1, 3)
    rgb_out = rgb.copy().reshape(-1, 3)
    points_out[bg] = 0
    rgb_out[bg] = 0
    return (
        points_out.reshape(H, W, 3),
        rgb_out.reshape(H, W, 3),
        fg.reshape(H, W),
    )


def _normalise_points(points: np.ndarray) -> np.ndarray:
    """Zero-mean and unit-radius normalisation (typical for PointNet++)."""
    valid = np.linalg.norm(points, axis=1) > 1e-6
    if valid.sum() < 3:
        return points
    centroid = points[valid].mean(axis=0)
    centered = points - centroid
    scale = np.linalg.norm(centered[valid], axis=1).max()
    if scale < 1e-6:
        return centered
    return centered / scale


def _load_and_prepare(
    rgb_path: Path,
    xyz_path: Path,
    img_size: int,
    ransac_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an image + organised point cloud, resize and remove the background.

    Returns
    -------
    rgb : (H, W, 3) uint8 image
    xyz : (H, W, 3) float32 organised point cloud, normalised to unit radius
    fg_mask : (H, W) bool foreground mask
    """
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    xyz = tifffile.imread(str(xyz_path)).astype(np.float32)
    if xyz.ndim == 2:
        xyz = np.stack([xyz, xyz, xyz], axis=-1)
    if xyz.shape[:2] != rgb.shape[:2]:
        xyz = _resize_xyz(xyz, max(rgb.shape[0], rgb.shape[1]))
    rgb = _resize_image(rgb, img_size)
    xyz = _resize_xyz(xyz, img_size)
    xyz, rgb, fg_mask = _remove_background(xyz, rgb, distance=ransac_distance)
    xyz = _normalise_points(xyz.reshape(-1, 3)).reshape(img_size, img_size, 3)
    return rgb, xyz, fg_mask


def _sample_points_random(
    points: np.ndarray,
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy fallback: sample ``num_points`` foreground pixels independently
    per sample. Kept for backward compatibility; **not** suitable for the
    per-point Gaussian bank used by 2M3DF.
    """
    H, W, _ = points.shape
    fg_flat = fg_mask.reshape(-1)
    pts_flat = points.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    fg_idx = np.where(fg_flat)[0]
    if fg_idx.size == 0:
        fg_idx = np.arange(H * W)
    if fg_idx.size >= num_points:
        sel = rng.choice(fg_idx, num_points, replace=False)
    else:
        sel = rng.choice(fg_idx, num_points, replace=True)
    pts = pts_flat[sel].astype(np.float32)
    cols = rgb_flat[sel].astype(np.float32) / 255.0
    v = (sel // W).astype(np.int64)
    u = (sel % W).astype(np.int64)
    return pts, cols, np.stack([u, v], axis=1)


def _extract_at_anchors(
    points: np.ndarray,
    rgb: np.ndarray,
    anchors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-point (xyz, color) at fixed pixel positions ``anchors``.

    Parameters
    ----------
    points : (H, W, 3) float32
    rgb    : (H, W, 3) uint8
    anchors : (N, 2) int64 (u, v) pixel positions.

    Returns
    -------
    pts : (N, 3) float32
    cols : (N, 3) float32 in [0, 1]
    pix_idx : (N, 2) int64 — same as ``anchors``.
    """
    u = anchors[:, 0].astype(np.int64)
    v = anchors[:, 1].astype(np.int64)
    pts = points[v, u].astype(np.float32)
    cols = rgb[v, u].astype(np.float32) / 255.0
    return pts, cols, anchors.astype(np.int64)


def compute_shared_anchors(
    samples: List[Tuple[Path, Path, Optional[Path], int, str]],
    num_points: int,
    img_size: int,
    ransac_distance: float = 0.005,
    seed: int = 0,
    max_samples_for_vote: int = 50,
    fg_vote_threshold: float = 0.5,
) -> np.ndarray:
    """Compute ``num_points`` shared anchor pixel positions for a category.

    The procedure is:
        1. Pick up to ``max_samples_for_vote`` training samples.
        2. For each one, run :func:`_load_and_prepare` and compute the (H, W)
           foreground mask.
        3. Average all masks to obtain a per-pixel foreground probability map.
        4. Keep pixels whose probability is ``>= fg_vote_threshold``. If fewer
           than ``num_points`` pixels survive, fall back to all pixels with
           probability > 0.
        5. Sample ``num_points`` pixels deterministically (``np.random``
           seeded by ``seed``) from the surviving set.

    Returns
    -------
    np.ndarray
        ``(num_points, 2)`` int64 array of ``(u, v)`` pixel positions.
    """
    if len(samples) == 0:
        raise ValueError("no samples provided to compute_shared_anchors")
    rng = np.random.default_rng(seed)
    n = min(len(samples), max_samples_for_vote)
    if len(samples) > n:
        idx = rng.choice(len(samples), n, replace=False)
        chosen = [samples[i] for i in idx]
    else:
        chosen = list(samples)

    vote = np.zeros((img_size, img_size), dtype=np.float64)
    for rgb_path, xyz_path, _, _, _ in chosen:
        _, _, fg_mask = _load_and_prepare(rgb_path, xyz_path, img_size, ransac_distance)
        vote += fg_mask.astype(np.float64)
    vote /= float(n)

    reliable = vote >= fg_vote_threshold
    if reliable.sum() < num_points:
        reliable = vote > 0
    if reliable.sum() < num_points:
        reliable = np.ones_like(reliable, dtype=bool)

    rows, cols = np.where(reliable)
    fg_indices = np.stack([cols, rows], axis=1)
    if fg_indices.shape[0] >= num_points:
        sel = rng.choice(fg_indices.shape[0], num_points, replace=False)
    else:
        sel = rng.choice(fg_indices.shape[0], num_points, replace=True)
    anchors = fg_indices[sel].astype(np.int64)
    return anchors


class MVTec3DCategoryDataset(Dataset):
    """Single-category MVTec3D-AD dataset.

    Parameters
    ----------
    root : str | Path
        Path to the dataset root (the folder containing per-category folders).
    category : str
        One of the 10 MVTec3D-AD categories.
    split : {"train", "test", "validation"}
        Which split to load.
    img_size : int
        Resize images and organised point clouds to this resolution.
    num_points : int
        Number of points to sample per cloud (foreground after RANSAC).
        Ignored when ``anchors`` is provided.
    anchors : np.ndarray or None
        Optional ``(N, 2)`` int array of ``(u, v)`` pixel positions shared by
        every sample. **Strongly recommended** — see module docstring. When
        ``None`` the legacy random-per-sample path is used.
    ransac_distance : float
        Plane-distance threshold for RANSAC background removal (Sec. IV-B).
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        img_size: int = 224,
        num_points: int = 4096,
        anchors: Optional[np.ndarray] = None,
        ransac_distance: float = 0.005,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if split not in {"train", "test", "validation"}:
            raise ValueError(f"unknown split: {split}")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.img_size = img_size
        self.num_points = num_points if anchors is None else int(anchors.shape[0])
        self.ransac_distance = ransac_distance
        self.anchors = None if anchors is None else np.asarray(anchors, dtype=np.int64)
        self._rng = np.random.default_rng(seed)

        self.samples: List[Tuple[Path, Path, Optional[Path], int, str]] = []
        cat_dir = self.root / category / split
        if not cat_dir.exists():
            raise FileNotFoundError(cat_dir)

        for defect_dir in sorted(cat_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect = defect_dir.name
            label = 0 if defect == "good" else 1
            rgb_dir = defect_dir / "rgb"
            xyz_dir = defect_dir / "xyz"
            gt_dir = defect_dir / "gt"
            if not rgb_dir.exists() or not xyz_dir.exists():
                continue
            for rgb_path in sorted(rgb_dir.iterdir()):
                stem = rgb_path.stem
                xyz_path = xyz_dir / f"{stem}.tiff"
                if not xyz_path.exists():
                    continue
                gt_path = gt_dir / f"{stem}.png" if gt_dir.exists() else None
                if gt_path is not None and not gt_path.exists():
                    gt_path = None
                self.samples.append((rgb_path, xyz_path, gt_path, label, defect))

        if not self.samples:
            raise RuntimeError(f"No samples found for {category}/{split} under {self.root}")

    def set_anchors(self, anchors: np.ndarray) -> None:
        """Attach a pre-computed anchor array to this dataset."""
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.num_points = int(self.anchors.shape[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, xyz_path, gt_path, label, defect = self.samples[idx]
        rgb, xyz, fg_mask = _load_and_prepare(
            rgb_path, xyz_path, self.img_size, self.ransac_distance
        )

        if self.anchors is not None:
            pts, cols, pix_idx = _extract_at_anchors(xyz, rgb, self.anchors)
        else:
            pts, cols, pix_idx = _sample_points_random(
                xyz, rgb, fg_mask, self.num_points, self._rng
            )

        mask = None
        if gt_path is not None:
            gt = np.array(Image.open(gt_path).convert("L"))
            gt = _resize_mask(gt, self.img_size)
            mask = torch.from_numpy((gt > 127).astype(np.uint8))

        return {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0),
            "points": torch.from_numpy(pts),
            "colors": torch.from_numpy(cols),
            "pixel_index": torch.from_numpy(pix_idx),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask if mask is not None else torch.zeros(self.img_size, self.img_size, dtype=torch.uint8),
            "has_mask": torch.tensor(mask is not None),
            "category": self.category,
            "defect": defect,
            "path": str(rgb_path),
        }


class MVTec3DAD:
    """Convenience class enumerating MVTec3D-AD categories and building datasets."""

    CATEGORIES = _DEFAULT_CATEGORIES

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    def build(
        self,
        category: str,
        split: str = "train",
        img_size: int = 224,
        num_points: int = 4096,
        anchors: Optional[np.ndarray] = None,
        ransac_distance: float = 0.005,
        seed: int = 0,
    ) -> MVTec3DCategoryDataset:
        return MVTec3DCategoryDataset(
            root=self.root,
            category=category,
            split=split,
            img_size=img_size,
            num_points=num_points,
            anchors=anchors,
            ransac_distance=ransac_distance,
            seed=seed,
        )
