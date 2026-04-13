"""Collators and connectors for RFDet3D pipeline.

Dramatically simplified compared to WildDet3D's 1500-line connector.
Standard per-image batching, no per-prompt expansion.
"""

from __future__ import annotations

import torch
from torch import Tensor

from wilddet3d_rfdetr.data_types import RFDet3DInput


class RFDet3DCollator:
    """Collate per-image dataset samples into RFDet3DInput batch.

    Handles variable-length GT annotations by keeping them as lists.
    """

    def __call__(self, batch: list[dict]) -> RFDet3DInput:
        images = torch.stack([b["images"] for b in batch])
        intrinsics = torch.stack([b["intrinsics"] for b in batch])

        # Variable-length GT (stay as lists)
        gt_boxes2d = None
        gt_boxes3d = None
        gt_labels = None

        if "boxes2d" in batch[0] and batch[0]["boxes2d"] is not None:
            gt_boxes2d = [b["boxes2d"] for b in batch]
        if "boxes3d" in batch[0] and batch[0]["boxes3d"] is not None:
            gt_boxes3d = [b["boxes3d"] for b in batch]
        if "boxes2d_classes" in batch[0] and batch[0]["boxes2d_classes"] is not None:
            gt_labels = [b["boxes2d_classes"] for b in batch]

        # Depth
        depth_gt = None
        depth_mask = None
        if "depth_maps" in batch[0] and batch[0]["depth_maps"] is not None:
            depth_maps = [b["depth_maps"] for b in batch]
            depth_gt = torch.stack(depth_maps, dim=0)
            if depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(1)

        # Metadata
        sample_names = [b.get("sample_names", "") for b in batch]
        dataset_name = [b.get("dataset_name", "") for b in batch]
        original_hw = [b.get("original_hw", images.shape[2:]) for b in batch]
        padding = [b.get("padding", [0, 0, 0, 0]) for b in batch]
        input_hw = [b.get("input_hw", images.shape[2:]) for b in batch]

        original_intrinsics = None
        if "original_intrinsics" in batch[0]:
            original_intrinsics = torch.stack(
                [b["original_intrinsics"] for b in batch]
            )

        return RFDet3DInput(
            images=images,
            intrinsics=intrinsics,
            gt_boxes2d=gt_boxes2d,
            gt_boxes3d=gt_boxes3d,
            gt_labels=gt_labels,
            depth_gt=depth_gt,
            depth_mask=depth_mask,
            sample_names=sample_names,
            dataset_name=dataset_name,
            original_hw=original_hw,
            original_intrinsics=original_intrinsics,
            padding=padding,
            input_hw=input_hw,
        )


class RFDet3DModelConnector:
    """Pass batch directly to model."""

    def __call__(self, data: RFDet3DInput) -> dict:
        return {"batch": data}


class RFDet3DLossConnector:
    """Route model output + batch to loss function."""

    def __call__(self, predictions, data: RFDet3DInput) -> dict:
        return {"out": predictions, "batch": data}


class RFDet3DEvalConnector:
    """Route model output to evaluator.

    In eval mode, model returns Det3DOut with per-image lists.
    """

    def __call__(self, predictions, data: RFDet3DInput) -> dict:
        return {
            "boxes": predictions.boxes,
            "boxes3d": predictions.boxes3d,
            "scores": predictions.scores,
            "class_ids": predictions.class_ids,
            "sample_names": data.sample_names,
            "dataset_name": data.dataset_name,
        }
