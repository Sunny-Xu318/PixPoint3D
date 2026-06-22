"""Top-level ``PixPoint3D`` network.

PixPoint3D is the main model of this repository — a multi-view, pixel-to-point
multimodal fusion network for 3D industrial anomaly detection (based on the
2M3DF paper by Asad et al., IEEE TCSVT 2025).

It glues together the four building blocks of the method:

    multi-view rendering -> 2D feature extraction -> point un-projection -> fusion
                          \
                           -> 3D point feature extraction -> fusion

The ``forward`` method returns the fused per-point features along with
auxiliary tensors useful for training (i.e. the un-projected 2D features and
the raw 3D features, both used by the cosine alignment loss).

By default the 2D backbone is frozen and PointNet++ is fine-tuned jointly with
the two fusion MLPs. Set ``freeze_pointnet=True`` (and ideally provide a
pre-trained PointNet++ checkpoint) to recover the paper's exact training
recipe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms

from .feature_extractor import PointFeatureExtractor, RGBFeatureExtractor
from .fusion import InterModalityFusion
from utils.multiview import generate_multi_view_renderings
from utils.projection import unproject_features_to_points


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PixPoint3DOutput:
    fused: torch.Tensor
    f_2d: torch.Tensor
    f_3d: torch.Tensor
    f_hat_2d: torch.Tensor
    f_hat_3d: torch.Tensor
    point_to_pixel: torch.Tensor
    valid_mask: torch.Tensor


class PixPoint3D(nn.Module):
    """Multi-View Pixel-to-Point Fusion network for 3D anomaly detection."""

    def __init__(
        self,
        d_2d: int = 512,
        d_3d: int = 128,
        img_size: int = 224,
        num_views: int = 7,
        backbone_2d: str = "wide_resnet50_2",
        layer_2d: str = "layer3",
        rot_x_range: Tuple[float, float] = (-90.0, 90.0),
        rot_y_range: Tuple[float, float] = (-10.0, 10.0),
        rot_z_range: Tuple[float, float] = (-10.0, 10.0),
        pretrained_pointnet: Optional[str] = None,
        freeze_pointnet: bool = False,
    ) -> None:
        super().__init__()
        self.d_2d = d_2d
        self.d_3d = d_3d
        self.img_size = img_size
        self.num_views = num_views
        self.rot_x_range = rot_x_range
        self.rot_y_range = rot_y_range
        self.rot_z_range = rot_z_range

        self.rgb_extractor = RGBFeatureExtractor(
            backbone=backbone_2d,
            layer=layer_2d,
            out_size=img_size,
            project_dim=d_2d,
        )
        self.point_extractor = PointFeatureExtractor(
            out_dim=d_3d,
            pretrained_ckpt=pretrained_pointnet,
            freeze=freeze_pointnet,
        )
        self.fusion = InterModalityFusion(d_2d=d_2d, d_3d=d_3d)
        self.normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    @property
    def fused_dim(self) -> int:
        return self.d_2d + self.d_3d

    def _normalise_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalise a batch of rendered images in [0, 1]."""
        mean = torch.tensor(_IMAGENET_MEAN, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
        return (imgs - mean) / std

    def render(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        rotations: Optional[torch.Tensor] = None,
    ):
        """Generate ``num_views`` colorised renderings for a single point cloud."""
        return generate_multi_view_renderings(
            points,
            colors,
            img_size=self.img_size,
            num_views=self.num_views,
            rot_x_range=self.rot_x_range,
            rot_y_range=self.rot_y_range,
            rot_z_range=self.rot_z_range,
            intrinsics=intrinsics,
            generator=generator,
            rotations=rotations,
        )

    def forward(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        rotations: Optional[torch.Tensor] = None,
    ) -> PixPoint3DOutput:
        """Run the full PixPoint3D pipeline on a single sample.

        Parameters
        ----------
        points : Tensor
            (N, 3) point cloud.
        colors : Tensor
            (N, 3) per-point RGB colour in [0, 1].
        intrinsics : Tensor, optional
            (3, 3) camera intrinsics. If ``None`` an orthographic fallback is
            used.
        rotations : Tensor, optional
            ``(num_views, 3, 3)`` pre-computed rotation matrices used to skip
            the random angle sampling. Pass this at inference time so that the
            view set is identical to the one used to fit the Gaussian bank.

        Returns
        -------
        PixPoint3DOutput
        """
        if points.dim() != 2 or points.shape[1] != 3:
            raise ValueError(f"`points` should be (N, 3), got {tuple(points.shape)}")

        rendering = self.render(
            points, colors,
            intrinsics=intrinsics,
            generator=generator,
            rotations=rotations,
        )
        images = self._normalise_images(rendering.images)
        with torch.no_grad():
            feature_maps = self.rgb_extractor(images)

        f_2d = unproject_features_to_points(
            feature_maps, rendering.point_to_pixel, rendering.valid_mask
        )
        f_3d = self.point_extractor(points.unsqueeze(0))[0]

        f_hat_2d, f_hat_3d, fused = self.fusion(
            f_2d.unsqueeze(0), f_3d.unsqueeze(0)
        )
        return PixPoint3DOutput(
            fused=fused.squeeze(0),
            f_2d=f_2d,
            f_3d=f_3d,
            f_hat_2d=f_hat_2d.squeeze(0),
            f_hat_3d=f_hat_3d.squeeze(0),
            point_to_pixel=rendering.point_to_pixel,
            valid_mask=rendering.valid_mask,
        )

    def trainable_parameters(self):
        """Yield the parameters that should be updated by the optimiser.

        The inter-modality MLPs are *always* trainable; PointNet++ is added
        only when ``freeze_pointnet=False`` was passed at construction (i.e.
        when no pre-trained checkpoint is available).
        """
        params = list(self.fusion.parameters())
        if not getattr(self.point_extractor, "_frozen", True):
            params.extend(self.point_extractor.parameters())
        return params
