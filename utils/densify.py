"""Anchor-to-pixel anomaly-score densification.

The Gaussian bank in 2M3DF produces one anomaly score per *anchor* point,
typically a few thousand values whereas the localisation evaluation
(AUPRO / segmentation AUROC) is computed on the full ``H x W`` pixel grid.

This module turns a sparse ``(N,)`` score vector into a dense ``(H, W)`` map
using K-nearest-neighbour inverse-distance weighting followed by an optional
Gaussian smoothing, both fully vectorised with NumPy / SciPy.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def densify_score_map(
    scores: np.ndarray,
    anchors: np.ndarray,
    img_size: int,
    k: int = 4,
    smooth_sigma: float = 4.0,
    power: float = 2.0,
) -> np.ndarray:
    """Turn ``N`` per-anchor anomaly scores into an ``(H, W)`` image.

    Parameters
    ----------
    scores : (N,) float array
        Anomaly score for each anchor point (e.g. Mahalanobis distance).
    anchors : (N, 2) int array
        Pixel positions ``(u, v)`` of the anchor points.
    img_size : int
        Side length of the output square image. The map is returned at
        ``(img_size, img_size)``.
    k : int
        Number of nearest neighbours used to interpolate each pixel. ``k=1``
        gives a nearest-neighbour assignment.
    smooth_sigma : float
        Standard deviation of the optional Gaussian post-filter (in pixels).
        Set to ``0`` to disable smoothing.
    power : float
        Exponent on the inverse distance. Larger values produce sharper maps.

    Returns
    -------
    np.ndarray
        ``(img_size, img_size)`` dense score map.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    if scores.shape[0] != anchors.shape[0]:
        raise ValueError(
            f"`scores` and `anchors` must have matching length: {scores.shape[0]} vs {anchors.shape[0]}"
        )

    H = W = int(img_size)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixels = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

    tree = cKDTree(anchors)
    k_eff = max(1, min(k, anchors.shape[0]))
    dists, idx = tree.query(pixels, k=k_eff)
    if k_eff == 1:
        dense = scores[idx]
    else:
        weights = 1.0 / (dists ** power + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        dense = (scores[idx] * weights).sum(axis=1)

    dense = dense.reshape(H, W)
    if smooth_sigma > 0:
        dense = gaussian_filter(dense, sigma=smooth_sigma)
    return dense
