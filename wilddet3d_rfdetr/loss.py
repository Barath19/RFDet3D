"""RFDet3DLoss: Combined 2D (RF-DETR) + 3D loss.

Uses RF-DETR's SetCriterion for 2D losses (focal + L1 + GIoU) and
ports WildDet3D's 3D losses (delta_center, depth, dims, rotation)
with shared matching indices.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wilddet3d_rfdetr._setup_paths import *  # noqa: F401, F403
from wilddet3d_rfdetr.data_types import RFDet3DInput, RFDet3DOut

from wilddet3d.head.coder_3d import Det3DCoder
from wilddet3d.ops.box2d import bbox_cxcywh_to_xyxy


class RFDet3DLoss(nn.Module):
    """Combined 2D + 3D loss for RFDet3D.

    2D losses come from RF-DETR's SetCriterion (focal + L1 + GIoU).
    3D losses are ported from WildDet3D (delta_center, depth, dims, rotation).
    Both share the same Hungarian matching indices.
    """

    def __init__(
        self,
        num_classes: int = 80,
        box_coder: Det3DCoder | None = None,
        # 2D loss config
        matcher_cost_class: float = 2.0,
        matcher_cost_bbox: float = 5.0,
        matcher_cost_giou: float = 2.0,
        cls_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        focal_alpha: float = 0.25,
        # 3D loss config
        loss_3d_scale: float = 1.0,
        loss_delta_2d_weight: float = 1.0,
        loss_depth_weight: float = 1.0,
        loss_dim_weight: float = 1.0,
        loss_rot_weight: float = 1.0,
        # 3D confidence
        loss_3d_conf_weight: float = 0.0,
        # Geometry backend
        loss_geom_scale: float = 5.0,
        # RF-DETR specific
        group_detr: int = 13,
        num_decoder_layers: int = 3,
        ia_bce_loss: bool = True,
    ) -> None:
        super().__init__()

        self.box_coder = box_coder or Det3DCoder()
        self.num_classes = num_classes
        self.loss_3d_scale = loss_3d_scale
        self.loss_delta_2d_weight = loss_delta_2d_weight
        self.loss_depth_weight = loss_depth_weight
        self.loss_dim_weight = loss_dim_weight
        self.loss_rot_weight = loss_rot_weight
        self.loss_3d_conf_weight = loss_3d_conf_weight
        self.loss_geom_scale = loss_geom_scale

        # Build RF-DETR matcher and criterion
        from rfdetr.models.matcher import HungarianMatcher
        from rfdetr.models.criterion import SetCriterion

        matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
            focal_alpha=focal_alpha,
        )

        # Weight dict for 2D losses
        weight_dict = {
            "loss_ce": cls_loss_coef,
            "loss_bbox": bbox_loss_coef,
            "loss_giou": giou_loss_coef,
        }
        # Aux layer weights
        for i in range(num_decoder_layers - 1):
            weight_dict.update({f"{k}_{i}": v for k, v in
                                {"loss_ce": cls_loss_coef,
                                 "loss_bbox": bbox_loss_coef,
                                 "loss_giou": giou_loss_coef}.items()})
        # Encoder weights
        weight_dict.update({f"{k}_enc": v for k, v in
                            {"loss_ce": cls_loss_coef,
                             "loss_bbox": bbox_loss_coef,
                             "loss_giou": giou_loss_coef}.items()})

        self.criterion_2d = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=focal_alpha,
            losses=["labels", "boxes"],
            group_detr=group_detr,
            ia_bce_loss=ia_bce_loss,
        )

    def forward(
        self,
        out: RFDet3DOut,
        batch: RFDet3DInput,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Compute all losses.

        Args:
            out: Model training output.
            batch: Input batch with GT annotations.

        Returns:
            Dict of weighted loss tensors.
        """
        device = out.pred_logits.device
        B = out.pred_logits.shape[0]
        H, W = batch.images.shape[2:]

        # ========== 1. Build RF-DETR targets ==========
        targets = self._build_targets(batch, device)

        # ========== 2. RF-DETR 2D losses ==========
        rfdetr_outputs = {
            "pred_logits": out.pred_logits,
            "pred_boxes": out.pred_boxes_2d,
        }
        if out.aux_outputs is not None:
            rfdetr_outputs["aux_outputs"] = [
                {"pred_logits": a["pred_logits"], "pred_boxes": a["pred_boxes_2d"]}
                for a in out.aux_outputs
            ]

        losses_2d = self.criterion_2d(rfdetr_outputs, targets)

        # Apply weight dict
        losses = {}
        for k, v in losses_2d.items():
            if k in self.criterion_2d.weight_dict:
                losses[k] = v * self.criterion_2d.weight_dict[k]
            elif not k.startswith("loss_"):
                # Metrics (class_error, etc.) — keep unweighted
                losses[k] = v

        # ========== 3. 3D losses ==========
        if out.pred_boxes_3d is not None and batch.gt_boxes3d is not None:
            # Run matcher for final layer to get indices for 3D
            group_detr = self.criterion_2d.group_detr if self.training else 1
            outputs_no_aux = {
                "pred_logits": out.pred_logits,
                "pred_boxes": out.pred_boxes_2d,
            }
            indices = self.criterion_2d.matcher(
                outputs_no_aux, targets, group_detr=group_detr
            )

            loss_3d = self._loss_boxes_3d(
                out.pred_boxes_2d, out.pred_boxes_3d,
                indices, targets, batch.intrinsics,
                image_size=(H, W),
            )
            for k, v in loss_3d.items():
                losses[k] = self.loss_3d_scale * v

        # ========== 4. 3D confidence loss ==========
        if self.loss_3d_conf_weight > 0 and out.pred_conf_3d is not None:
            # Simple BCE: positive targets for matched, negative for unmatched
            losses["loss_3d_conf"] = self.loss_3d_conf_weight * self._loss_3d_conf(
                out.pred_conf_3d, indices, len(targets)
            )

        # ========== 5. Geometry backend losses ==========
        if out.geom_losses:
            for k, v in out.geom_losses.items():
                if isinstance(v, Tensor) and v.requires_grad:
                    losses[f"geom_{k}"] = self.loss_geom_scale * v

        return losses

    def _build_targets(
        self, batch: RFDet3DInput, device: torch.device
    ) -> list[dict]:
        """Convert batch GT to RF-DETR target format.

        RF-DETR expects: list of dicts with "labels" (N_i,) and "boxes" (N_i, 4) cxcywh normalized.
        """
        targets = []
        B = batch.images.shape[0]
        _, _, H, W = batch.images.shape

        for i in range(B):
            if batch.gt_labels is not None and batch.gt_boxes2d is not None:
                labels = batch.gt_labels[i].to(device)
                boxes = batch.gt_boxes2d[i].to(device)

                # Convert pixel xyxy to normalized cxcywh
                boxes_norm = boxes.clone()
                boxes_norm[:, 0::2] /= W
                boxes_norm[:, 1::2] /= H
                # xyxy to cxcywh
                cx = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2
                cy = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2
                w = boxes_norm[:, 2] - boxes_norm[:, 0]
                h = boxes_norm[:, 3] - boxes_norm[:, 1]
                boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)

                target_dict = {"labels": labels, "boxes": boxes_cxcywh}
                # Pass through 3D GT if available
                if batch.gt_boxes3d is not None and batch.gt_boxes3d[i] is not None:
                    target_dict["boxes3d"] = batch.gt_boxes3d[i].to(device)
                targets.append(target_dict)
            else:
                targets.append({
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                    "boxes": torch.zeros(0, 4, device=device),
                })

        return targets

    def _loss_boxes_3d(
        self,
        pred_boxes_2d: Tensor,
        pred_boxes_3d: Tensor,
        indices: list[tuple[Tensor, Tensor]],
        targets: list[dict],
        intrinsics: Tensor,
        image_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Compute 3D box regression losses.

        Ported from wilddet3d.loss.wilddet3d_loss.WildDet3DLoss._loss_boxes_3d.
        Uses GT 2D boxes for encoding targets (stable), predicted 3D params for loss.
        """
        device = pred_boxes_3d.device
        H, W = image_size

        # Collect matched predictions and targets across batch
        all_src_3d = []
        all_tgt_3d_encoded = []
        all_weights_3d = []

        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            # Predicted 3D params for matched queries
            src_3d = pred_boxes_3d[i][src_idx]  # (M, 12)

            # GT 2D boxes in pixel coords for encoding
            gt_boxes_xyxy = targets[i]["boxes"][tgt_idx]  # (M, 4) normalized cxcywh
            gt_boxes_xyxy = bbox_cxcywh_to_xyxy(gt_boxes_xyxy)
            gt_boxes_pixel = gt_boxes_xyxy.clone()
            gt_boxes_pixel[:, 0::2] *= W
            gt_boxes_pixel[:, 1::2] *= H

            # GT 3D boxes
            gt_3d = self._get_gt_boxes3d(i, tgt_idx, targets, device)
            if gt_3d is None:
                continue

            # Encode GT 3D → target parameterization
            K = intrinsics[i]  # (3, 3)
            for j in range(len(src_idx)):
                single_3d = gt_3d[j:j+1]
                if single_3d.abs().sum() < 1e-6:
                    continue
                single_box_pixel = gt_boxes_pixel[j:j+1]
                encoded, weights = self.box_coder.encode(
                    single_box_pixel, single_3d, K
                )
                all_src_3d.append(src_3d[j:j+1])
                all_tgt_3d_encoded.append(encoded)
                all_weights_3d.append(weights)

        if len(all_src_3d) == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "loss_delta_2d": zero, "loss_depth": zero,
                "loss_dim": zero, "loss_rot": zero,
            }

        src_3d = torch.cat(all_src_3d, dim=0)
        tgt_3d = torch.cat(all_tgt_3d_encoded, dim=0)
        weights = torch.cat(all_weights_3d, dim=0)
        num_boxes = max(len(all_src_3d), 1)

        losses = {}

        # Delta 2D center
        losses["loss_delta_2d"] = self.loss_delta_2d_weight * (
            F.l1_loss(src_3d[:, :2], tgt_3d[:, :2], reduction="none")
            * weights[:, :2]
        ).sum() / num_boxes

        # Depth
        losses["loss_depth"] = self.loss_depth_weight * (
            F.l1_loss(src_3d[:, 2:3], tgt_3d[:, 2:3], reduction="none")
            * weights[:, 2:3]
        ).sum() / num_boxes

        # Dimensions
        losses["loss_dim"] = self.loss_dim_weight * (
            F.l1_loss(src_3d[:, 3:6], tgt_3d[:, 3:6], reduction="none")
            * weights[:, 3:6]
        ).sum() / num_boxes

        # Rotation
        losses["loss_rot"] = self.loss_rot_weight * (
            F.l1_loss(src_3d[:, 6:], tgt_3d[:, 6:], reduction="none")
            * weights[:, 6:]
        ).sum() / num_boxes

        return losses

    def _get_gt_boxes3d(
        self, batch_idx: int, tgt_idx: Tensor,
        targets: list[dict], device: torch.device,
    ) -> Tensor | None:
        """Get GT 3D boxes for matched targets."""
        if "boxes3d" not in targets[batch_idx]:
            return None
        gt_3d = targets[batch_idx]["boxes3d"]
        if gt_3d is None or len(gt_3d) == 0:
            return None
        return gt_3d[tgt_idx].to(device)

    def _loss_3d_conf(
        self,
        pred_conf: Tensor,
        indices: list[tuple[Tensor, Tensor]],
        batch_size: int,
    ) -> Tensor:
        """Simple BCE loss for 3D confidence."""
        device = pred_conf.device
        B, Q, _ = pred_conf.shape
        target = torch.zeros(B, Q, 1, device=device)

        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target[i, src_idx] = 1.0

        return F.binary_cross_entropy_with_logits(pred_conf, target)
