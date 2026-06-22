"""Multivariate Gaussian normality bank used by 2M3DF (Sec. III.I-J).

Each spatial point ``p`` stores ``(mu_p, Sigma_p)`` estimated from all training
fused features, with regularisation ``Sigma_p += alpha * I`` (Eq. 13).

At inference time we expose the Mahalanobis distance (Eq. 14)
``D_M(x_p) = sqrt((x - mu_p)^T Sigma_p^{-1} (x - mu_p))`` as the per-point
anomaly score.

The implementation uses two-pass online estimators (Welford-style) so that the
bank scales to many training samples without keeping a giant tensor in memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class MultivariateGaussianBank:
    """Per-point multivariate Gaussian (mu, Sigma) bank.

    Usage:
        bank = MultivariateGaussianBank(num_points=N, feat_dim=D)
        for fused_features in loader:
            bank.update(fused_features)
        bank.finalise(alpha=0.09)
        scores = bank.mahalanobis(test_features)  # (N,)
    """

    def __init__(self, num_points: int, feat_dim: int, device: str | torch.device = "cpu") -> None:
        self.num_points = num_points
        self.feat_dim = feat_dim
        self.device = torch.device(device)
        self.count = 0
        self.mean = torch.zeros(num_points, feat_dim, device=self.device)
        self.M2 = torch.zeros(num_points, feat_dim, feat_dim, device=self.device)
        self.inv_cov: Optional[torch.Tensor] = None
        self._finalised = False

    def update(self, features: torch.Tensor) -> None:
        """Accumulate one batch of fused features.

        Parameters
        ----------
        features : Tensor
            ``(B, N, D)`` per-point features for B training samples.
        """
        if self._finalised:
            raise RuntimeError("Bank already finalised, cannot update.")
        if features.dim() == 2:
            features = features.unsqueeze(0)
        x = features.to(self.device)
        B, N, D = x.shape
        if N != self.num_points or D != self.feat_dim:
            raise ValueError(
                f"Expected features (*, {self.num_points}, {self.feat_dim}), got {tuple(x.shape)}"
            )
        for b in range(B):
            xb = x[b]
            self.count += 1
            delta = xb - self.mean
            self.mean = self.mean + delta / self.count
            delta2 = xb - self.mean
            self.M2 = self.M2 + torch.einsum("ni,nj->nij", delta, delta2)

    def finalise(self, alpha: float = 0.09) -> None:
        """Finalise the bank: compute Sigma = M2/(n-1) + alpha*I and its inverse."""
        if self._finalised:
            return
        n = max(self.count - 1, 1)
        cov = self.M2 / n
        eye = torch.eye(self.feat_dim, device=self.device).expand_as(cov)
        cov = cov + alpha * eye
        L = torch.linalg.cholesky(cov)
        eye_b = torch.eye(self.feat_dim, device=self.device).expand_as(cov)
        inv = torch.cholesky_solve(eye_b, L)
        self.inv_cov = inv
        self._finalised = True

    def mahalanobis(self, features: torch.Tensor) -> torch.Tensor:
        """Compute the per-point Mahalanobis distance for one sample.

        Parameters
        ----------
        features : Tensor
            ``(N, D)`` fused features of a test sample.

        Returns
        -------
        Tensor
            ``(N,)`` Mahalanobis distance per point.
        """
        if not self._finalised:
            raise RuntimeError("Bank must be finalised before calling mahalanobis().")
        x = features.to(self.device)
        diff = x - self.mean
        m = torch.einsum("ni,nij,nj->n", diff, self.inv_cov, diff)
        m = torch.clamp(m, min=0.0)
        return torch.sqrt(m + 1e-8)

    def state_dict(self) -> dict:
        return {
            "num_points": self.num_points,
            "feat_dim": self.feat_dim,
            "count": self.count,
            "mean": self.mean.cpu(),
            "M2": self.M2.cpu(),
            "inv_cov": None if self.inv_cov is None else self.inv_cov.cpu(),
            "finalised": self._finalised,
        }

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "MultivariateGaussianBank":
        state = torch.load(str(path), map_location="cpu")
        bank = cls(state["num_points"], state["feat_dim"], device=device)
        bank.count = state["count"]
        bank.mean = state["mean"].to(device)
        bank.M2 = state["M2"].to(device)
        if state["inv_cov"] is not None:
            bank.inv_cov = state["inv_cov"].to(device)
        bank._finalised = state["finalised"]
        return bank

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), str(path))
