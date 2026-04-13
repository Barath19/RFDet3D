"""RFDet3D Training Script.

Trains the 3D detection head on top of a frozen RF-DETR 2D detector.
Supports wandb logging and checkpoint saving.

Usage:
    python train.py --rfdetr_variant base --epochs 24 --batch_size 4 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import wandb

from wilddet3d_rfdetr import RFDet3D, RFDet3DInput
from wilddet3d_rfdetr.loss import RFDet3DLoss
from wilddet3d_rfdetr.connector import RFDet3DCollator
from wilddet3d_rfdetr.inference import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCO3DDataset(Dataset):
    """Simple dataset for images with 3D bounding box annotations.

    Expects a directory structure:
        data_root/
            annotations.json     # COCO-style with 3D extensions
            images/              # image files

    annotations.json format:
    {
        "images": [{"id": 1, "file_name": "xxx.jpg", "width": W, "height": H,
                     "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]]}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [x,y,w,h],
                         "bbox3d": {"center": [cx,cy,cz], "dimensions": [w,l,h],
                                    "quaternion": [qw,qx,qy,qz]}}],
        "categories": [{"id": 1, "name": "person"}]
    }

    If no annotations.json is found, falls back to image-only mode
    (for inference/visualization without GT).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        target_size: int = 560,
        max_depth: float = 100.0,
    ):
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.max_depth = max_depth

        ann_file = self.data_root / f"{split}_annotations.json"
        if not ann_file.exists():
            ann_file = self.data_root / "annotations.json"

        if ann_file.exists():
            with open(ann_file) as f:
                self.coco = json.load(f)
            self.images = {img["id"]: img for img in self.coco["images"]}
            self.image_ids = list(self.images.keys())

            # Group annotations by image
            self.img_to_anns = {}
            for ann in self.coco.get("annotations", []):
                img_id = ann["image_id"]
                self.img_to_anns.setdefault(img_id, []).append(ann)

            # Category mapping to 0-based indices
            self.cat_to_idx = {
                cat["id"]: i for i, cat in enumerate(self.coco["categories"])
            }
        else:
            # Image-only mode
            img_dir = self.data_root / "images"
            img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            self.images = {i: {"id": i, "file_name": f.name} for i, f in enumerate(img_files)}
            self.image_ids = list(self.images.keys())
            self.img_to_anns = {}
            self.cat_to_idx = {}

        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.data_root / "images" / img_info["file_name"]
        from PIL import Image
        image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        H_orig, W_orig = image.shape[:2]

        # Intrinsics
        if "intrinsics" in img_info:
            intrinsics = np.array(img_info["intrinsics"], dtype=np.float32)
        else:
            focal = max(H_orig, W_orig)
            intrinsics = np.array([
                [focal, 0, W_orig / 2],
                [0, focal, H_orig / 2],
                [0, 0, 1],
            ], dtype=np.float32)

        # Resize
        import cv2
        scale_x = self.target_size / W_orig
        scale_y = self.target_size / H_orig
        image = cv2.resize(image / 255.0, (self.target_size, self.target_size))
        intrinsics[0, :] *= scale_x
        intrinsics[1, :] *= scale_y

        # Normalize
        image = (image - self.mean) / self.std
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        intrinsics_tensor = torch.from_numpy(intrinsics).float()

        # Annotations
        anns = self.img_to_anns.get(img_id, [])
        boxes2d = []
        boxes3d = []
        labels = []

        for ann in anns:
            # 2D box: COCO [x,y,w,h] → pixel xyxy, then scale
            x, y, w, h = ann["bbox"]
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes2d.append([x1, y1, x2, y2])

            # Category
            cat_idx = self.cat_to_idx.get(ann["category_id"], 0)
            labels.append(cat_idx)

            # 3D box
            if "bbox3d" in ann:
                b3d = ann["bbox3d"]
                center = b3d.get("center", [0, 0, 0])
                dims = b3d.get("dimensions", [0, 0, 0])
                quat = b3d.get("quaternion", [1, 0, 0, 0])
                boxes3d.append(center + dims + quat)
            else:
                boxes3d.append([0] * 10)

        if len(boxes2d) > 0:
            boxes2d = torch.tensor(boxes2d, dtype=torch.float32)
            boxes3d = torch.tensor(boxes3d, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes2d = torch.zeros(0, 4, dtype=torch.float32)
            boxes3d = torch.zeros(0, 10, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        # Depth map (if available)
        depth_path = self.data_root / "depth" / img_info["file_name"].replace(".jpg", ".npz").replace(".png", ".npz")
        depth_map = None
        if depth_path.exists():
            depth_data = np.load(depth_path)
            depth_map = depth_data.get("depth", depth_data[list(depth_data.keys())[0]])
            depth_map = cv2.resize(depth_map, (self.target_size, self.target_size))
            depth_map = np.clip(depth_map, 0, self.max_depth)
            depth_map = torch.from_numpy(depth_map).float()

        return {
            "images": image_tensor,
            "intrinsics": intrinsics_tensor,
            "boxes2d": boxes2d,
            "boxes3d": boxes3d,
            "boxes2d_classes": labels,
            "depth_maps": depth_map,
            "sample_names": img_info["file_name"],
            "original_hw": (H_orig, W_orig),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def build_optimizer(model: RFDet3D, args) -> torch.optim.Optimizer:
    """Build optimizer with per-component learning rates."""
    param_groups = []

    # RF-DETR: frozen in phase 1, low LR in phase 2
    rfdetr_lr = 0.0 if args.phase == 1 else args.lr * 0.01
    rfdetr_params = [p for n, p in model.named_parameters() if "rfdetr_model" in n and p.requires_grad]
    if rfdetr_params:
        param_groups.append({"params": rfdetr_params, "lr": rfdetr_lr, "name": "rfdetr"})

    # 3D head: full LR
    head_params = [p for n, p in model.named_parameters() if "bbox3d_head" in n and p.requires_grad]
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr, "name": "bbox3d_head"})

    # Geometry backend: 0.1x LR
    geom_params = [p for n, p in model.named_parameters() if "geometry_backend" in n and p.requires_grad]
    if geom_params:
        param_groups.append({"params": geom_params, "lr": args.lr * 0.1, "name": "geometry_backend"})

    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def train_one_epoch(
    model: RFDet3D,
    loss_fn: RFDet3DLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    args,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        batch = _to_device(batch, device)

        # Forward
        out = model(batch)
        losses = loss_fn(out, batch)

        # Total loss
        grad_losses = [v for v in losses.values() if isinstance(v, torch.Tensor) and v.requires_grad]
        if grad_losses:
            loss = sum(grad_losses)
        else:
            # All losses are detached (e.g., 3D targets not available).
            # Create a dummy loss connected to trainable params for DDP compat.
            loss = sum(p.sum() * 0 for p in model.parameters() if p.requires_grad)
            loss = loss + sum(v for v in losses.values() if isinstance(v, torch.Tensor)).detach()

        # Backward
        optimizer.zero_grad()
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()

        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log
        if batch_idx % args.log_interval == 0:
            log_dict = {f"train/{k}": v.item() for k, v in losses.items()
                        if isinstance(v, torch.Tensor)}
            log_dict["train/loss"] = loss.item()
            log_dict["train/lr"] = optimizer.param_groups[-1]["lr"]
            log_dict["train/epoch"] = epoch
            log_dict["train/step"] = epoch * len(dataloader) + batch_idx
            wandb.log(log_dict)

            if batch_idx % (args.log_interval * 10) == 0:
                print(f"  [{batch_idx}/{len(dataloader)}] loss={loss.item():.4f} "
                      f"ce={losses.get('loss_ce', torch.tensor(0)).item():.4f} "
                      f"depth={losses.get('loss_depth', torch.tensor(0)).item():.4f}")

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def _to_device(batch: RFDet3DInput, device: torch.device) -> RFDet3DInput:
    """Move batch tensors to device."""
    batch.images = batch.images.to(device)
    batch.intrinsics = batch.intrinsics.to(device)
    if batch.gt_boxes2d is not None:
        batch.gt_boxes2d = [b.to(device) for b in batch.gt_boxes2d]
    if batch.gt_boxes3d is not None:
        batch.gt_boxes3d = [b.to(device) for b in batch.gt_boxes3d]
    if batch.gt_labels is not None:
        batch.gt_labels = [b.to(device) for b in batch.gt_labels]
    if batch.depth_gt is not None:
        batch.depth_gt = batch.depth_gt.to(device)
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RFDet3D Training")

    # Model
    parser.add_argument("--rfdetr_variant", type=str, default="base",
                        choices=["nano", "small", "base", "medium", "large"])
    parser.add_argument("--num_classes", type=int, default=80)
    parser.add_argument("--wilddet3d_ckpt", type=str, default=None,
                        help="Path to WildDet3D checkpoint for 3D head weight transfer")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root with annotations.json and images/")
    parser.add_argument("--val_data_root", type=str, default=None)
    parser.add_argument("--target_size", type=int, default=560)

    # Training
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=0.1)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1: freeze RF-DETR. Phase 2: fine-tune all.")
    parser.add_argument("--num_workers", type=int, default=4)

    # Loss
    parser.add_argument("--loss_3d_scale", type=float, default=1.0)
    parser.add_argument("--loss_geom_scale", type=float, default=5.0)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="rfdet3d")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)

    # Checkpoints
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--save_interval", type=int, default=4,
                        help="Save checkpoint every N epochs")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # ---- Setup ----
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # ---- wandb ----
    run_name = args.wandb_name or f"rfdet3d-{args.rfdetr_variant}-phase{args.phase}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    # ---- Model ----
    print(f"\nBuilding RFDet3D ({args.rfdetr_variant}, {args.num_classes} classes, phase {args.phase})...")
    freeze_rfdetr = args.phase == 1
    model = RFDet3D(
        rfdetr_variant=args.rfdetr_variant,
        num_classes=args.num_classes,
        freeze_rfdetr=freeze_rfdetr,
        score_threshold=0.3,
    )

    if args.wilddet3d_ckpt:
        model.load_pretrained(wilddet3d_ckpt=args.wilddet3d_ckpt)

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    wandb.config.update({"trainable_params": trainable, "total_params": total})

    # ---- Data ----
    print(f"\nLoading dataset from {args.data_root}...")
    train_dataset = COCO3DDataset(
        data_root=args.data_root,
        split="train",
        target_size=args.target_size,
    )
    print(f"Training samples: {len(train_dataset)}")

    collator = RFDet3DCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Loss ----
    # Get num decoder layers from RF-DETR config
    num_dec_layers = len(model.rfdetr_model.transformer.decoder.layers) if hasattr(model.rfdetr_model.transformer, 'decoder') else 3
    loss_fn = RFDet3DLoss(
        num_classes=args.num_classes,
        box_coder=model.box_coder,
        loss_3d_scale=args.loss_3d_scale,
        loss_geom_scale=args.loss_geom_scale,
        num_decoder_layers=num_dec_layers,
    )
    loss_fn = loss_fn.to(device)

    # ---- Optimizer ----
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 2 / 3), int(args.epochs * 5 / 6)],
        gamma=0.1,
    )

    # ---- Train ----
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Phase {args.phase}: {'RF-DETR frozen' if freeze_rfdetr else 'All trainable'}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print("-" * 60)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        avg_loss = train_one_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            device, epoch, args,
        )

        elapsed = time.time() - t0
        print(f"  avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")
        wandb.log({"epoch/avg_loss": avg_loss, "epoch/time_s": elapsed, "epoch": epoch + 1})

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or avg_loss < best_loss:
            ckpt_path = os.path.join(args.output_dir, f"rfdet3d_epoch{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, "rfdet3d_best.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "loss": avg_loss,
                }, best_path)
                print(f"  Best model saved: {best_path}")

    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
