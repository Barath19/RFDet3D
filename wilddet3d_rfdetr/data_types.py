"""Data types for RFDet3D pipeline.

Simplified compared to WildDet3D — no per-prompt batching, no geometry prompts.
Standard per-image batch format for closed-vocabulary detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from torch import Tensor


@dataclass
class RFDet3DInput:
    """Batched input for RFDet3D. Standard image-batch format.

    Unlike WildDet3D's per-prompt batching (N_prompts, img_ids, text_ids),
    this uses simple per-image batching: B images produce B sets of detections.
    """

    # Image-level data
    images: Tensor  # (B, 3, H, W) ImageNet-normalized
    intrinsics: Tensor  # (B, 3, 3) camera matrices

    # Ground truth (training only) — variable-length per image
    gt_boxes2d: list[Tensor] | None = None  # list of (N_i, 4) pixel xyxy
    gt_boxes3d: list[Tensor] | None = None  # list of (N_i, 10) [center(3), dims(3), quat(4)]
    gt_labels: list[Tensor] | None = None  # list of (N_i,) class indices (0-based)

    # Depth (optional — ARKit LiDAR, stereo, etc.)
    depth_gt: Tensor | None = None  # (B, 1, H, W) depth in meters
    depth_mask: Tensor | None = None  # (B, H, W) valid depth mask

    # Metadata
    sample_names: list[str] | None = None
    dataset_name: list[str] | None = None
    original_hw: list[tuple[int, int]] | None = None
    original_intrinsics: Tensor | None = None  # (B, 3, 3) before resize/pad
    padding: list[list[int]] | None = None  # (B,) [left, right, top, bottom]
    input_hw: list[tuple[int, int]] | None = None  # (B,) after resize, before pad


class RFDet3DOut(NamedTuple):
    """Training output from RFDet3D model."""

    pred_logits: Tensor  # (B, num_queries, num_classes)
    pred_boxes_2d: Tensor  # (B, num_queries, 4) normalized cxcywh
    pred_boxes_3d: Tensor | None  # (B, num_queries, 12) encoded 3D params
    pred_conf_3d: Tensor | None  # (B, num_queries, 1) 3D confidence
    aux_outputs: list[dict] | None  # deep supervision from decoder layers
    geom_losses: dict[str, Tensor] | None  # geometry backend losses
