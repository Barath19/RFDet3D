"""Inference utilities for RFDet3D.

Provides a user-friendly predictor and build_model factory.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from wilddet3d_rfdetr._setup_paths import *  # noqa: F401, F403
from wilddet3d_rfdetr.data_types import RFDet3DInput
from wilddet3d_rfdetr.model import RFDet3D

from wilddet3d.data_types import Det3DOut
from wilddet3d.head.coder_3d import Det3DCoder


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(
    image: np.ndarray,
    intrinsics: np.ndarray | None = None,
    target_size: int = 560,
) -> dict:
    """Preprocess a single image for RFDet3D inference.

    Args:
        image: RGB image as numpy array (H, W, 3), float32 [0, 255] or uint8.
        intrinsics: Camera intrinsics (3, 3). If None, uses default
            (focal=max(H,W), principal point at center).
        target_size: Target image size (square). Matches RF-DETR resolution.

    Returns:
        Dict with preprocessed tensors ready for model input.
    """
    H_orig, W_orig = image.shape[:2]

    # Default intrinsics
    if intrinsics is None:
        focal = max(H_orig, W_orig)
        intrinsics = np.array([
            [focal, 0, W_orig / 2],
            [0, focal, H_orig / 2],
            [0, 0, 1],
        ], dtype=np.float32)

    # Normalize to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        image = image / 255.0

    # Resize to target_size x target_size
    import cv2
    scale_x = target_size / W_orig
    scale_y = target_size / H_orig
    image_resized = cv2.resize(image, (target_size, target_size))

    # Scale intrinsics
    intrinsics_scaled = intrinsics.copy()
    intrinsics_scaled[0, :] *= scale_x
    intrinsics_scaled[1, :] *= scale_y

    # ImageNet normalize
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    image_norm = (image_resized - mean) / std

    # To tensor (H, W, C) -> (C, H, W) -> (1, C, H, W)
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    intrinsics_tensor = torch.from_numpy(intrinsics_scaled).unsqueeze(0)

    return {
        "images": image_tensor,
        "intrinsics": intrinsics_tensor,
        "original_hw": (H_orig, W_orig),
        "input_hw": (target_size, target_size),
        "padding": [0, 0, 0, 0],
    }


class RFDet3DPredictor(torch.nn.Module):
    """User-friendly inference wrapper for RFDet3D."""

    def __init__(self, model: RFDet3D):
        super().__init__()
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        intrinsics: Tensor,
        original_hw: list[tuple[int, int]] | None = None,
        input_hw: list[tuple[int, int]] | None = None,
        padding: list[list[int]] | None = None,
        depth_gt: Optional[Tensor] = None,
    ) -> Det3DOut:
        """Run inference.

        Args:
            images: (B, 3, H, W) ImageNet-normalized.
            intrinsics: (B, 3, 3) camera matrices.
            original_hw: Original image sizes for rescaling boxes.
            depth_gt: Optional (B, 1, H, W) depth in meters.

        Returns:
            Det3DOut with per-image lists of boxes, boxes3d, scores, class_ids.
        """
        batch = RFDet3DInput(
            images=images,
            intrinsics=intrinsics,
            original_hw=original_hw,
            input_hw=input_hw,
            padding=padding,
            depth_gt=depth_gt,
        )
        return self.model(batch)


def build_model(
    rfdetr_variant: str = "base",
    num_classes: int = 80,
    checkpoint: str | None = None,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    device: str = "cpu",
) -> RFDet3DPredictor:
    """Build RFDet3D model for inference.

    Args:
        rfdetr_variant: RF-DETR variant ("nano", "small", "base", "medium", "large").
        num_classes: Number of detection classes (default 80 for COCO).
        checkpoint: Path to RFDet3D checkpoint with 3D head weights.
        score_threshold: Score threshold for filtering detections.
        device: Device to place model on.

    Returns:
        RFDet3DPredictor ready for inference.
    """
    model = RFDet3D(
        rfdetr_variant=rfdetr_variant,
        num_classes=num_classes,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )

    if checkpoint is not None:
        model.load_pretrained(wilddet3d_ckpt=checkpoint)

    model = model.to(device)
    model.eval()

    return RFDet3DPredictor(model)
