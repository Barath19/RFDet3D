"""Download and convert ARKitScenes 3DOD to RFDet3D training format.

Downloads the ARKitScenes 3D Object Detection split from Apple's CDN,
samples frames, projects 3D boxes into camera coordinates, and produces
COCO-style annotations with 3D extensions.

Usage:
    # Download validation only (small, ~60GB) for testing
    python scripts/prepare_arkitscenes.py --split Validation --output_dir data/arkitscenes_val

    # Download training (large, ~560GB)
    python scripts/prepare_arkitscenes.py --split Training --output_dir data/arkitscenes_train

    # Download both
    python scripts/prepare_arkitscenes.py --split both --output_dir data/arkitscenes

    # Limit scenes for quick test
    python scripts/prepare_arkitscenes.py --split Validation --max_scenes 10 --output_dir data/arkit_test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# ARKitScenes 17 classes
# ---------------------------------------------------------------------------

ARKITSCENES_CLASSES = [
    "cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer",
    "toilet", "bathtub", "oven", "dishwasher", "fireplace", "stool",
    "chair", "table", "tv_monitor", "sofa",
]

CLASS_TO_ID = {name: i for i, name in enumerate(ARKITSCENES_CLASSES)}

# Labels that map to our classes (handle spaces/hyphens)
LABEL_NORMALIZE = {
    "tv monitor": "tv_monitor",
    "tv-monitor": "tv_monitor",
    "tv_monitor": "tv_monitor",
}

CDN_BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/threedod"
SPLITS_CSV_URL = f"{CDN_BASE}/../threedod/3dod_train_val_splits.csv"

# Scenes with missing assets
MISSING_SCENES = {
    "47334522", "47334523", "42897421", "45261582", "47333152", "47333155",
    "48458535", "48018733", "47429677", "48458541", "42897848", "47895482",
    "47333960", "47430089", "42899148", "42897612", "42899153", "42446164",
    "48018149", "47332198", "47334515", "45663223", "45663226", "45663227",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> bool:
    """Download a file with curl. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    print(f"  Downloading {url}")
    result = subprocess.run(
        ["curl", "-L", "-o", str(dest), "--fail", "--silent", "--show-error", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}")
        if dest.exists():
            dest.unlink()
        return False
    return True


def get_scene_list(split: str, download_dir: Path) -> list[tuple[str, str]]:
    """Get list of (video_id, split) from the official CSV."""
    csv_path = download_dir / "3dod_train_val_splits.csv"

    if not csv_path.exists():
        # Download the CSV
        csv_url = f"{CDN_BASE}/3dod_train_val_splits.csv"
        print(f"Downloading scene list from {csv_url}")
        if not download_file(csv_url, csv_path):
            # Try alternate URL
            csv_url2 = "https://raw.githubusercontent.com/apple/ARKitScenes/main/threedod/3dod_train_val_splits.csv"
            download_file(csv_url2, csv_path)

    if not csv_path.exists():
        print("ERROR: Could not download scene list CSV. Creating from known splits.")
        return []

    scenes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "").strip()
            fold = row.get("fold", "").strip()
            if not vid or not fold:
                continue
            if vid in MISSING_SCENES:
                continue
            if split == "both" or fold == split:
                scenes.append((vid, fold))

    return scenes


def download_scene(video_id: str, split: str, download_dir: Path) -> Path | None:
    """Download and extract a single scene. Returns scene dir or None."""
    scene_dir = download_dir / "3dod" / split / video_id
    if scene_dir.exists() and (scene_dir / f"{video_id}_3dod_annotation.json").exists():
        return scene_dir

    zip_url = f"{CDN_BASE}/{split}/{video_id}.zip"
    zip_path = download_dir / "zips" / f"{video_id}.zip"

    if not download_file(zip_url, zip_path):
        return None

    # Extract
    print(f"  Extracting {video_id}...")
    extract_dir = download_dir / "3dod" / split
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        print(f"  BAD ZIP: {zip_path}")
        zip_path.unlink()
        return None

    # Clean up zip to save disk
    zip_path.unlink()

    if scene_dir.exists():
        return scene_dir
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def load_intrinsics(pincam_path: Path) -> np.ndarray:
    """Load a .pincam file → (3, 3) intrinsic matrix."""
    vals = np.loadtxt(pincam_path)
    w, h, fx, fy, cx, cy = vals[:6]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def load_trajectory(traj_path: Path) -> dict[str, np.ndarray]:
    """Load camera trajectory → dict of timestamp → 4x4 world-to-camera matrix."""
    poses = {}
    with open(traj_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 7:
                continue
            ts = tokens[0]
            angle_axis = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            translation = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])])

            R, _ = cv2.Rodrigues(angle_axis)
            extrinsic = np.eye(4, dtype=np.float64)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = translation
            poses[ts] = extrinsic

    return poses


def load_3d_annotations(ann_path: Path) -> list[dict]:
    """Load 3D annotations from scene JSON."""
    with open(ann_path) as f:
        data = json.load(f)

    if data.get("skipped", False):
        return []

    boxes = []
    for obj in data.get("data", []):
        label = obj.get("label", "").strip().lower()
        # Normalize label
        label = LABEL_NORMALIZE.get(label, label.replace(" ", "_").replace("-", "_"))

        if label not in CLASS_TO_ID:
            continue

        obb = obj.get("segments", {}).get("obbAligned", {})
        if not obb:
            continue

        centroid = np.array(obb["centroid"], dtype=np.float64)
        axes_lengths = np.array(obb["axesLengths"], dtype=np.float64)
        rot_matrix = np.array(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3)

        boxes.append({
            "label": label,
            "category_id": CLASS_TO_ID[label],
            "centroid_world": centroid,
            "dimensions": axes_lengths,  # full sizes, not half
            "rotation_world": rot_matrix,
        })

    return boxes


def project_box_to_camera(
    centroid_world: np.ndarray,
    dimensions: np.ndarray,
    rotation_world: np.ndarray,
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    img_w: int,
    img_h: int,
) -> dict | None:
    """Project a 3D world-space box into camera coordinates.

    Returns dict with camera-space center, dims, quaternion, and 2D bbox,
    or None if the box is behind the camera or fully outside the image.
    """
    # Transform center to camera coordinates
    center_world_h = np.append(centroid_world, 1.0)
    center_cam_h = extrinsic_w2c @ center_world_h
    center_cam = center_cam_h[:3]

    # Skip if behind camera
    if center_cam[2] <= 0.1:
        return None

    # Rotation: world → camera
    R_w2c = extrinsic_w2c[:3, :3]
    rot_cam = R_w2c @ rotation_world

    # Convert rotation to quaternion
    rot_obj = Rotation.from_matrix(rot_cam)
    quat = rot_obj.as_quat()  # [x, y, z, w] scipy convention
    quat_wxyz = [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]

    # Compute 2D bounding box by projecting 8 corners
    half = dimensions / 2.0
    corners_local = np.array([
        [-half[0], -half[1], -half[2]],
        [half[0], -half[1], -half[2]],
        [half[0], half[1], -half[2]],
        [-half[0], half[1], -half[2]],
        [-half[0], -half[1], half[2]],
        [half[0], -half[1], half[2]],
        [half[0], half[1], half[2]],
        [-half[0], half[1], half[2]],
    ])

    # Rotate and translate corners to camera space
    corners_cam = (rot_cam @ corners_local.T).T + center_cam

    # Skip if any corner is behind camera
    if (corners_cam[:, 2] <= 0).any():
        return None

    # Project to image
    corners_2d = (intrinsic @ corners_cam.T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]

    x1 = max(0, float(corners_2d[:, 0].min()))
    y1 = max(0, float(corners_2d[:, 1].min()))
    x2 = min(img_w, float(corners_2d[:, 0].max()))
    y2 = min(img_h, float(corners_2d[:, 1].max()))

    # Skip tiny or invalid boxes
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    # Skip if mostly outside image
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    if box_area / img_area < 0.001 or x1 >= img_w * 0.95 or y1 >= img_h * 0.95:
        return None

    return {
        "center": center_cam.tolist(),
        "dimensions": dimensions.tolist(),
        "quaternion": quat_wxyz,
        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format [x, y, w, h]
    }


# ---------------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------------

def process_scene(
    scene_dir: Path,
    video_id: str,
    output_dir: Path,
    frame_step: int = 10,
    max_frames_per_scene: int = 50,
) -> tuple[list[dict], list[dict]]:
    """Process one ARKitScenes scene → COCO images + annotations.

    Returns (images_list, annotations_list).
    """
    ann_path = scene_dir / f"{video_id}_3dod_annotation.json"
    traj_path = scene_dir / f"{video_id}_frames" / "lowres_wide.traj"
    rgb_dir = scene_dir / f"{video_id}_frames" / "lowres_wide"
    depth_dir = scene_dir / f"{video_id}_frames" / "lowres_depth"
    intrinsics_dir = scene_dir / f"{video_id}_frames" / "lowres_wide_intrinsics"

    if not ann_path.exists() or not traj_path.exists() or not rgb_dir.exists():
        return [], []

    # Load scene data
    boxes_3d = load_3d_annotations(ann_path)
    if not boxes_3d:
        return [], []

    poses = load_trajectory(traj_path)

    # Get frame list
    rgb_files = sorted(rgb_dir.glob(f"{video_id}_*.png"))
    if not rgb_files:
        return [], []

    # Sample frames
    rgb_files = rgb_files[::frame_step][:max_frames_per_scene]

    images_list = []
    anns_list = []
    img_dir = output_dir / "images"
    depth_out_dir = output_dir / "depth"
    img_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)

    for rgb_path in rgb_files:
        # Extract timestamp
        fname = rgb_path.stem  # video_id_timestamp
        ts = fname.replace(f"{video_id}_", "")

        # Find matching intrinsics
        pincam_path = intrinsics_dir / f"{fname}.pincam"
        if not pincam_path.exists():
            continue

        # Find matching pose (closest timestamp)
        if ts not in poses:
            # Try matching with float comparison
            try:
                ts_float = float(ts)
                best_ts = min(poses.keys(), key=lambda t: abs(float(t) - ts_float))
                if abs(float(best_ts) - ts_float) > 0.1:
                    continue
                pose = poses[best_ts]
            except (ValueError, TypeError):
                continue
        else:
            pose = poses[ts]

        intrinsic = load_intrinsics(pincam_path)

        # Read image to get dimensions
        img = cv2.imread(str(rgb_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Project all 3D boxes into this frame
        frame_anns = []
        for box in boxes_3d:
            proj = project_box_to_camera(
                box["centroid_world"],
                box["dimensions"],
                box["rotation_world"],
                pose,
                intrinsic,
                img_w, img_h,
            )
            if proj is not None:
                frame_anns.append({
                    "category_id": box["category_id"],
                    "label": box["label"],
                    **proj,
                })

        if not frame_anns:
            continue

        # Copy image
        out_fname = f"{video_id}_{ts}.jpg"
        out_path = img_dir / out_fname
        if not out_path.exists():
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Copy depth (if exists)
        depth_path = depth_dir / f"{fname}.png"
        if depth_path.exists():
            depth_out_path = depth_out_dir / f"{video_id}_{ts}.npz"
            if not depth_out_path.exists():
                depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_img is not None:
                    depth_meters = depth_img.astype(np.float32) / 1000.0
                    np.savez_compressed(str(depth_out_path), depth=depth_meters)

        # Build COCO image entry
        img_entry = {
            "file_name": out_fname,
            "width": img_w,
            "height": img_h,
            "intrinsics": intrinsic.tolist(),
            "video_id": video_id,
            "timestamp": ts,
        }
        images_list.append(img_entry)

        # Build annotation entries (image_id set to index within this scene,
        # will be replaced with global ID by caller)
        frame_img_idx = len(images_list) - 1  # index of the image we just added
        for ann in frame_anns:
            anns_list.append({
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "bbox3d": {
                    "center": ann["center"],
                    "dimensions": ann["dimensions"],
                    "quaternion": ann["quaternion"],
                },
                "_local_img_idx": frame_img_idx,
            })

    return images_list, anns_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and convert ARKitScenes 3DOD")
    parser.add_argument("--split", type=str, default="Validation",
                        choices=["Training", "Validation", "both"],
                        help="Which split to download")
    parser.add_argument("--output_dir", type=str, default="data/arkitscenes",
                        help="Output directory for converted dataset")
    parser.add_argument("--download_dir", type=str, default="data/arkitscenes_raw",
                        help="Temp directory for raw downloads")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Limit number of scenes (for testing)")
    parser.add_argument("--frame_step", type=int, default=10,
                        help="Sample every Nth frame per scene")
    parser.add_argument("--max_frames_per_scene", type=int, default=50,
                        help="Max frames to sample per scene")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, only run conversion")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    # ---- Get scene list ----
    print("=" * 60)
    print("ARKitScenes 3DOD → RFDet3D Converter")
    print("=" * 60)

    scenes = get_scene_list(args.split, download_dir)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    print(f"Split: {args.split}")
    print(f"Scenes: {len(scenes)}")
    print(f"Frame step: {args.frame_step}")
    print(f"Max frames/scene: {args.max_frames_per_scene}")
    print()

    # ---- Download and process ----
    all_images = []
    all_annotations = []
    img_id_counter = 0
    ann_id_counter = 0

    for i, (video_id, split) in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] Scene {video_id} ({split})")

        # Download
        if not args.skip_download:
            scene_dir = download_scene(video_id, split, download_dir)
        else:
            scene_dir = download_dir / "3dod" / split / video_id

        if scene_dir is None or not scene_dir.exists():
            print(f"  Skipped (download failed)")
            continue

        # Process
        images, anns = process_scene(
            scene_dir, video_id, output_dir,
            frame_step=args.frame_step,
            max_frames_per_scene=args.max_frames_per_scene,
        )

        # Assign global IDs and link annotations to images
        img_id_base = img_id_counter
        for img in images:
            img["id"] = img_id_counter
            img_id_counter += 1

        for ann in anns:
            ann["id"] = ann_id_counter
            ann["image_id"] = img_id_base + ann.pop("_local_img_idx")
            ann_id_counter += 1

        all_images.extend(images)
        all_annotations.extend(anns)

        if images:
            print(f"  → {len(images)} frames, {len(anns)} annotations")

    # ---- Write output ----
    split_name = args.split.lower() if args.split != "both" else "all"
    categories = [{"id": i, "name": name} for i, name in enumerate(ARKITSCENES_CLASSES)]

    output = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories,
    }

    ann_file = output_dir / f"{split_name}_annotations.json"
    with open(ann_file, "w") as f:
        json.dump(output, f)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Images: {len(all_images)}")
    print(f"  Annotations: {len(all_annotations)}")
    print(f"  Classes: {len(ARKITSCENES_CLASSES)}")
    print(f"  Output: {ann_file}")
    print(f"  Images dir: {output_dir / 'images'}")
    print(f"  Depth dir: {output_dir / 'depth'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
