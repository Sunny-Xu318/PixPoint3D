"""Inter-modality feature representation learning + point-wise fusion module.

Section III.G of the paper. Two lightweight MLPs translate features back and
forth between the 2D and 3D modalities:

    phi_3d_to_2d : F_PC^{3D} -> hat F_MV^{2D}
    phi_2d_to_3d : F_MV^{2D} -> hat F_PC^{3D}

They are trained jointly with a cosine-similarity loss (Eq. 9) that asks the
predicted features to align with the ones extracted from the other modality.
At fusion time the *predicted* features are L2-normalised and concatenated
point-wise to form ``F_MM in R^{N x (d_2D + d_3D)}`` (Eq. 11).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    """Four fully-connected layers with ReLU between (no activation on last)."""

    def __init__(self, dims) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InterModalityFusion(nn.Module):
    """Cross-domain MLPs + cosine-similarity loss + point-wise concat fusion.

    Parameters
    ----------
    d_2d : int
        Dimensionality of the 2D RGB per-point features. Default 512.
    d_3d : int
        Dimensionality of the 3D point-cloud per-point features. Default 128.
    """

    def __init__(self, d_2d: int = 512, d_3d: int = 128) -> None:
        super().__init__()
        self.d_2d = d_2d
        self.d_3d = d_3d
        self.phi_3d_to_2d = MLPProjector([d_3d, 256, 512, d_2d])
        self.phi_2d_to_3d = MLPProjector([d_2d, 512, 256, d_3d])

    def forward(
        self, f_2d: torch.Tensor, f_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the two MLPs and return the predicted features + fused output.

        Parameters
        ----------
        f_2d : Tensor
            ``(B, N, d_2d)`` aggregated multi-view RGB point-wise features.
        f_3d : Tensor
            ``(B, N, d_3d)`` PointNet++ per-point features.

        Returns
        -------
        f_hat_2d : Tensor
            ``(B, N, d_2d)`` 2D features predicted from the 3D modality.
        f_hat_3d : Tensor
            ``(B, N, d_3d)`` 3D features predicted from the 2D modality.
        fused : Tensor
            ``(B, N, d_2d + d_3d)`` point-wise concatenated multimodal features
            (after L2 normalisation of each branch).
        """
        f_hat_2d = self.phi_3d_to_2d(f_3d)
        f_hat_3d = self.phi_2d_to_3d(f_2d)

        n_hat_3d = F.normalize(f_hat_3d, p=2, dim=-1)
        n_hat_2d = F.normalize(f_hat_2d, p=2, dim=-1)
        fused = torch.cat([n_hat_3d, n_hat_2d], dim=-1)
        return f_hat_2d, f_hat_3d, fused

    @staticmethod
    def cosine_alignment_loss(
        f_2d: torch.Tensor,
        f_hat_2d: torch.Tensor,
        f_3d: torch.Tensor,
        f_hat_3d: torch.Tensor,
    ) -> torch.Tensor:
        """Inter-modality alignment loss (Eq. 9).

        ``L_im = (1 - cos(F_MV_2D, hat F_MV_2D)) + (1 - cos(F_PC_3D, hat F_PC_3D))``
        averaged over points and batch.
        """
        cos_2d = F.cosine_similarity(f_2d, f_hat_2d, dim=-1)
        cos_3d = F.cosine_similarity(f_3d, f_hat_3d, dim=-1)
        loss = (1.0 - cos_2d).mean() + (1.0 - cos_3d).mean()
        return loss
