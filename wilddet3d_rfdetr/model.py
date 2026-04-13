"""RFDet3D: RF-DETR with 3D Detection Head.

Combines RF-DETR (2D detection) with WildDet3D's 3D detection head
and LingBot-Depth geometry backend. Apache 2.0 licensed alternative
to the SAM3-based WildDet3D.

Architecture:
    Image + Intrinsics
         |
    +----+----+
    |         |
  RF-DETR   LingBot-Depth
  (2D det)  (depth est.)
    |         |
    +----+----+
         |
    Det3DHead
    (depth cross-attn)
         |
    3D Boxes
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import batched_nms

from wilddet3d_rfdetr._setup_paths import *  # noqa: F401, F403
from wilddet3d_rfdetr.data_types import RFDet3DInput, RFDet3DOut

from wilddet3d.head.head_3d import Det3DHead
from wilddet3d.head.coder_3d import Det3DCoder
from wilddet3d.depth.base import GeometryBackendBase
from wilddet3d.data_types import Det3DOut
from wilddet3d.ops.box2d import bbox_cxcywh_to_xyxy


class RFDet3D(nn.Module):
    """RF-DETR with 3D Detection Head.

    This model combines:
    1. RF-DETR (DINOv2 backbone + deformable transformer decoder) for 2D detection
    2. LingBot-Depth geometry backend for monocular depth estimation
    3. Det3DHead for 3D box regression from decoder query embeddings

    Key differences from WildDet3D:
    - Closed-vocabulary (fixed class set, no text prompts)
    - Standard per-image batching (no per-prompt expansion)
    - Apache 2.0 license (RF-DETR) vs SAM License (SAM3)
    """

    def __init__(
        self,
        # RF-DETR
        rfdetr_variant: str = "base",
        num_classes: int = 80,
        # 3D Components (reused from wilddet3d)
        bbox3d_head: Det3DHead | None = None,
        box_coder: Det3DCoder | None = None,
        geometry_backend: GeometryBackendBase | None = None,
        # Freeze
        freeze_rfdetr: bool = False,
        # Eval settings
        score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5,
        eval_3d_conf_weight: float = 0.5,
        use_depth_input_test: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.eval_3d_conf_weight = eval_3d_conf_weight
        self.use_depth_input_test = use_depth_input_test

        # Build RF-DETR
        self.rfdetr_model, self.rfdetr_resolution = self._build_rfdetr(
            rfdetr_variant, num_classes
        )
        self.hidden_dim = self.rfdetr_model.transformer.d_model  # 256

        # Register forward hook to capture decoder hidden states
        self._last_hs = None
        self._register_hs_hook()

        # 3D components
        if bbox3d_head is None:
            bbox3d_head = Det3DHead(
                embed_dims=self.hidden_dim,
                box_coder=box_coder or Det3DCoder(),
            )
        self.bbox3d_head = bbox3d_head
        self.box_coder = box_coder or Det3DCoder()
        self.geometry_backend = geometry_backend

        # Freeze RF-DETR if requested
        if freeze_rfdetr:
            for param in self.rfdetr_model.parameters():
                param.requires_grad = False

    def _build_rfdetr(
        self, variant: str, num_classes: int
    ) -> tuple[nn.Module, int]:
        """Build RF-DETR model from variant name.

        Returns (model, resolution).
        """
        from rfdetr.detr import _build_model_context

        config_map = {
            "nano": "RFDETRNanoConfig",
            "small": "RFDETRSmallConfig",
            "base": "RFDETRBaseConfig",
            "medium": "RFDETRMediumConfig",
            "large": "RFDETRLargeConfig",
        }

        import rfdetr.config as cfg_module

        config_cls = getattr(cfg_module, config_map[variant])
        config = config_cls()
        config.num_classes = num_classes

        ctx = _build_model_context(config)
        return ctx.model, config.resolution

    def _register_hs_hook(self):
        """Register a forward hook on RF-DETR's transformer to capture
        decoder hidden states before class/box heads consume them."""

        def hook_fn(module, input, output):
            hs, ref_unsigmoid, hs_enc, ref_enc = output
            self._last_hs = hs  # (L, B, Q, 256) or (L, B, Q*G, 256) during training

        self.rfdetr_model.transformer.register_forward_hook(hook_fn)

    def forward(
        self, batch: RFDet3DInput
    ) -> RFDet3DOut | Det3DOut:
        """Forward pass.

        Args:
            batch: RFDet3DInput with images, intrinsics, and optional GT.

        Returns:
            RFDet3DOut in training mode, Det3DOut in eval mode.
        """
        B, _, H, W = batch.images.shape
        device = batch.images.device

        # ========== Step 1: Geometry backend (depth estimation) ==========
        geom_losses = None
        depth_latents = None
        geom_out = None

        if self.geometry_backend is not None:
            depth_gt = None
            depth_mask = None
            if self.training or self.use_depth_input_test:
                depth_gt = batch.depth_gt
            if self.training:
                depth_mask = batch.depth_mask

            geom_out = self.geometry_backend(
                images=batch.images,
                depth_feats=None,
                intrinsics=batch.intrinsics,
                image_hw=(H, W),
                depth_gt=depth_gt,
                depth_mask=depth_mask,
                padding=batch.padding,
            )
            depth_latents = geom_out.get("depth_latents")
            if self.training:
                geom_losses = geom_out.get("losses", {})

        # ========== Step 2: RF-DETR forward (2D detection) ==========
        # RF-DETR expects ImageNet-normalized images at its resolution
        images_rfdetr = self._prepare_images(batch.images)
        rfdetr_out = self.rfdetr_model(images_rfdetr)

        # Hidden states captured by hook: (L, B, Q, 256)
        hs = self._last_hs
        assert hs is not None, "Forward hook did not capture hidden states"

        # During training with group_detr, only use first group's queries
        num_queries = self.rfdetr_model.num_queries
        if self.training and hs.shape[2] > num_queries:
            # group_detr: hs is (L, B, Q*G, 256), take first group
            hs_for_3d = hs[:, :, :num_queries, :]
        else:
            hs_for_3d = hs

        # ========== Step 3: Ray embeddings from intrinsics ==========
        ray_embeddings = None
        ray_intrinsics = batch.intrinsics
        ray_image_hw = (H, W)
        ray_downsample = 16

        if geom_out is not None:
            ray_intrinsics = geom_out.get("ray_intrinsics", ray_intrinsics)
            ray_image_hw = geom_out.get("ray_image_hw", ray_image_hw)
            ray_downsample = geom_out.get("ray_downsample", ray_downsample)

        ray_embeddings = self.bbox3d_head.get_camera_embeddings(
            ray_intrinsics, ray_image_hw, ray_downsample
        )

        # ========== Step 4: Det3DHead (3D box regression) ==========
        all_boxes_3d, all_conf_3d = self.bbox3d_head(
            hidden_states=hs_for_3d,
            ray_embeddings=ray_embeddings,
            depth_latents=depth_latents,
        )

        pred_boxes_3d = all_boxes_3d[-1]  # (B, Q, 12)
        pred_conf_3d = all_conf_3d[-1]  # (B, Q, 1)

        # ========== Step 5: Build output ==========
        if self.training:
            # Build aux_outputs with 3D predictions per decoder layer
            aux_outputs = None
            if "aux_outputs" in rfdetr_out:
                aux_outputs = []
                for i, aux in enumerate(rfdetr_out["aux_outputs"]):
                    aux_dict = {
                        "pred_logits": aux["pred_logits"],
                        "pred_boxes_2d": aux["pred_boxes"],
                    }
                    if i < len(all_boxes_3d) - 1:
                        aux_dict["pred_boxes_3d"] = all_boxes_3d[i]
                        aux_dict["pred_conf_3d"] = all_conf_3d[i]
                    aux_outputs.append(aux_dict)

            return RFDet3DOut(
                pred_logits=rfdetr_out["pred_logits"][:, :num_queries],
                pred_boxes_2d=rfdetr_out["pred_boxes"][:, :num_queries],
                pred_boxes_3d=pred_boxes_3d,
                pred_conf_3d=pred_conf_3d,
                aux_outputs=aux_outputs,
                geom_losses=geom_losses,
            )
        else:
            return self._forward_test(
                rfdetr_out, pred_boxes_3d, pred_conf_3d, batch, geom_out
            )

    def _prepare_images(self, images: Tensor) -> Tensor:
        """Resize images to RF-DETR's expected resolution.

        RF-DETR uses ImageNet normalization (same as our pipeline), so
        no color space conversion needed — just resize.
        """
        B, C, H, W = images.shape
        res = self.rfdetr_resolution
        if H != res or W != res:
            return F.interpolate(
                images, size=(res, res), mode="bilinear", align_corners=False
            )
        return images

    @torch.no_grad()
    def _forward_test(
        self,
        rfdetr_out: dict,
        pred_boxes_3d: Tensor,
        pred_conf_3d: Tensor,
        batch: RFDet3DInput,
        geom_out: dict | None,
    ) -> Det3DOut:
        """Post-process predictions for inference."""
        B = batch.images.shape[0]
        _, _, H, W = batch.images.shape
        device = batch.images.device
        num_queries = self.rfdetr_model.num_queries

        pred_logits = rfdetr_out["pred_logits"][:, :num_queries]  # (B, Q, C)
        pred_boxes_cxcywh = rfdetr_out["pred_boxes"][:, :num_queries]  # (B, Q, 4)

        # Convert to pixel xyxy
        pred_boxes_xyxy = bbox_cxcywh_to_xyxy(pred_boxes_cxcywh)
        pred_boxes_xyxy[..., 0::2] *= W
        pred_boxes_xyxy[..., 1::2] *= H

        # Class scores
        scores_all = pred_logits.sigmoid()  # (B, Q, C)

        # Combine with 3D confidence
        if pred_conf_3d is not None and self.eval_3d_conf_weight > 0:
            conf_3d = pred_conf_3d.sigmoid()
            scores_all = scores_all + self.eval_3d_conf_weight * conf_3d

        boxes_list, boxes3d_list, scores_list, class_ids_list = [], [], [], []
        depth_maps = None

        if geom_out is not None and "depth_map" in geom_out:
            depth_maps = [geom_out["depth_map"][i] for i in range(B)]

        for i in range(B):
            scores_i = scores_all[i]  # (Q, C)
            max_scores, class_ids = scores_i.max(dim=-1)  # (Q,), (Q,)
            boxes_i = pred_boxes_xyxy[i]  # (Q, 4)

            # Score threshold
            keep = max_scores > self.score_threshold
            if keep.sum() == 0:
                boxes_list.append(torch.zeros(0, 4, device=device))
                boxes3d_list.append(torch.zeros(0, 10, device=device))
                scores_list.append(torch.zeros(0, device=device))
                class_ids_list.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            filtered_boxes = boxes_i[keep]
            filtered_scores = max_scores[keep]
            filtered_classes = class_ids[keep]

            # NMS
            keep_nms = batched_nms(
                filtered_boxes, filtered_scores,
                filtered_classes, self.nms_iou_threshold
            )

            final_boxes = filtered_boxes[keep_nms]
            final_scores = filtered_scores[keep_nms]
            final_classes = filtered_classes[keep_nms]

            # Decode 3D boxes
            if pred_boxes_3d is not None:
                final_3d_params = pred_boxes_3d[i][keep][keep_nms]
                final_boxes3d = self.box_coder.decode(
                    final_boxes, final_3d_params, batch.intrinsics[i]
                )
            else:
                final_boxes3d = torch.zeros(len(final_boxes), 10, device=device)

            # Rescale to original image if padding was applied
            if batch.padding is not None and batch.original_hw is not None:
                pad = batch.padding[i]  # [left, right, top, bottom]
                orig_h, orig_w = batch.original_hw[i]
                if isinstance(pad, (list, tuple)) and len(pad) == 4:
                    pad_left, pad_right, pad_top, pad_bottom = pad
                    final_boxes[:, 0] -= pad_left
                    final_boxes[:, 2] -= pad_left
                    final_boxes[:, 1] -= pad_top
                    final_boxes[:, 3] -= pad_top

            boxes_list.append(final_boxes)
            boxes3d_list.append(final_boxes3d)
            scores_list.append(final_scores)
            class_ids_list.append(final_classes)

        return Det3DOut(
            boxes=boxes_list,
            boxes3d=boxes3d_list,
            scores=scores_list,
            class_ids=class_ids_list,
            depth_maps=depth_maps,
            categories=None,
        )

    def load_pretrained(
        self,
        wilddet3d_ckpt: str | None = None,
    ) -> None:
        """Load pretrained weights.

        RF-DETR weights are loaded during __init__ (auto-download from HF).
        This method loads the 3D head + geometry backend from a WildDet3D checkpoint.
        """
        if wilddet3d_ckpt is None:
            return

        print(f"Loading 3D head weights from: {wilddet3d_ckpt}")
        ckpt = torch.load(wilddet3d_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Transfer 3D head weights
        head_state = {}
        geom_state = {}
        for key, value in state_dict.items():
            # Strip "model." prefix if present (vis4d convention)
            clean_key = key.replace("model.", "", 1) if key.startswith("model.") else key

            if clean_key.startswith("bbox3d_head."):
                head_state[clean_key.replace("bbox3d_head.", "")] = value
            elif clean_key.startswith("geometry_backend."):
                geom_state[clean_key.replace("geometry_backend.", "")] = value

        if head_state:
            missing, unexpected = self.bbox3d_head.load_state_dict(
                head_state, strict=False
            )
            print(f"  3D head: loaded {len(head_state)} params, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")

        if geom_state and self.geometry_backend is not None:
            missing, unexpected = self.geometry_backend.load_state_dict(
                geom_state, strict=False
            )
            print(f"  Geometry backend: loaded {len(geom_state)} params, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
