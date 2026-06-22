"""Evaluation metrics for industrial anomaly detection.

* ``compute_image_auroc``  -> image-level AUROC (I-AUROC).
* ``compute_aupro``        -> area under the per-region overlap (AUPRO) curve
  as originally proposed by Bergmann et al. (MVTec3D-AD paper). The default
  ``fpr_limit`` follows that work (0.3).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score


def compute_image_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Image-level AUROC (binary classification of anomalous vs. normal)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_aupro(
    masks: Sequence[np.ndarray],
    score_maps: Sequence[np.ndarray],
    fpr_limit: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """Area under the per-region overlap (PRO) curve.

    Parameters
    ----------
    masks : list of (H, W) binary ground-truth masks (1=anomaly).
    score_maps : list of (H, W) anomaly score maps.
    fpr_limit : float
        The FPR upper-bound used by the MVTec papers; the curve is normalised
        by ``fpr_limit`` so AUPRO@0.3 is in ``[0, 1]``.
    num_thresholds : int
        Number of thresholds at which to evaluate the PRO/FPR curve.

    Returns
    -------
    float
        AUPRO in ``[0, 1]`` (higher is better).
    """
    masks_arr = [np.asarray(m, dtype=bool) for m in masks]
    scores_arr = [np.asarray(s, dtype=np.float64) for s in score_maps]
    if len(masks_arr) == 0:
        return float("nan")

    flat_scores = np.concatenate([s.ravel() for s in scores_arr])
    flat_mask_neg = np.concatenate([(~m).ravel() for m in masks_arr])
    neg_scores = flat_scores[flat_mask_neg]
    if neg_scores.size == 0:
        return float("nan")

    s_min, s_max = flat_scores.min(), flat_scores.max()
    thresholds = np.linspace(s_max, s_min, num_thresholds)

    regions = []
    for m in masks_arr:
        lbl, num = label(m)
        comps = []
        for i in range(1, num + 1):
            comps.append(lbl == i)
        regions.append(comps)

    pros, fprs = [], []
    for t in thresholds:
        binarised = [s >= t for s in scores_arr]
        fp = sum((b & ~m).sum() for b, m in zip(binarised, masks_arr))
        n_neg = sum((~m).sum() for m in masks_arr)
        fpr = fp / max(n_neg, 1)
        overlaps = []
        for b, comps in zip(binarised, regions):
            for c in comps:
                if c.sum() == 0:
                    continue
                overlaps.append((b & c).sum() / c.sum())
        if not overlaps:
            continue
        pro = float(np.mean(overlaps))
        pros.append(pro)
        fprs.append(fpr)

    if not fprs:
        return float("nan")

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    keep = fprs <= fpr_limit
    if not keep.any():
        return 0.0
    fprs_kept = fprs[keep]
    pros_kept = pros[keep]
    if fprs_kept[-1] < fpr_limit:
        idx = np.searchsorted(fprs, fpr_limit)
        if idx < len(fprs):
            fprs_kept = np.concatenate([fprs_kept, [fpr_limit]])
            pros_kept = np.concatenate([pros_kept, [pros[idx]]])

    aupro = np.trapz(pros_kept, fprs_kept) / fpr_limit
    return float(np.clip(aupro, 0.0, 1.0))
