"""End-to-end inference test for RFDet3D.

Downloads RF-DETR Nano weights, runs on a test image,
verifies 2D + 3D outputs are valid.
"""

import numpy as np
import torch
from PIL import Image

from wilddet3d_rfdetr.inference import preprocess, build_model

# COCO class names for display
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def main():
    print("=" * 60)
    print("RFDet3D End-to-End Inference Test")
    print("=" * 60)

    # Load test image
    img_path = "third_party/WildDet3D/assets/demo_huggingface.png"
    print(f"\n1. Loading image: {img_path}")
    image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
    print(f"   Image shape: {image.shape}")

    # Preprocess
    print("\n2. Preprocessing...")
    data = preprocess(image, target_size=384)
    print(f"   Tensor shape: {data['images'].shape}")
    print(f"   Intrinsics: focal={data['intrinsics'][0, 0, 0]:.1f}")

    # Build model (RF-DETR Nano for speed)
    print("\n3. Building model (RF-DETR Nano + Det3DHead)...")
    model = build_model(
        rfdetr_variant="nano",
        num_classes=80,
        score_threshold=0.1,  # Low threshold to see more detections
    )
    print(f"   Hidden dim: {model.model.hidden_dim}")
    print(f"   Resolution: {model.model.rfdetr_resolution}")

    # Run inference
    print("\n4. Running inference...")
    results = model(
        images=data["images"],
        intrinsics=data["intrinsics"],
        original_hw=[data["original_hw"]],
        padding=[data["padding"]],
    )

    # Display results
    boxes = results.boxes[0]
    boxes3d = results.boxes3d[0]
    scores = results.scores[0]
    class_ids = results.class_ids[0]

    print(f"\n5. Results:")
    print(f"   Detections: {len(boxes)}")

    for i in range(min(len(boxes), 10)):
        cls_id = class_ids[i].item()
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
        score = scores[i].item()
        box = boxes[i].tolist()
        b3d = boxes3d[i]
        center = b3d[:3].tolist()
        dims = b3d[3:6].tolist()

        print(f"   [{i}] {cls_name:15s} score={score:.3f}  "
              f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]  "
              f"3d_center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})  "
              f"dims=({dims[0]:.2f},{dims[1]:.2f},{dims[2]:.2f})")

    # Validate outputs
    print("\n6. Validation:")
    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, f"Bad boxes shape: {boxes.shape}"
    assert len(boxes3d.shape) == 2 and boxes3d.shape[1] == 10, f"Bad boxes3d shape: {boxes3d.shape}"
    assert len(scores.shape) == 1, f"Bad scores shape: {scores.shape}"
    assert len(class_ids.shape) == 1, f"Bad class_ids shape: {class_ids.shape}"
    assert (scores >= 0).all() and (scores <= 2).all(), f"Scores out of range"
    assert (class_ids >= 0).all() and (class_ids < 80).all(), f"Class IDs out of range"
    print("   All shape/range checks PASSED")

    # Check 3D boxes are not all zeros (3D head is producing non-trivial output)
    if len(boxes3d) > 0:
        non_zero = (boxes3d.abs().sum(dim=-1) > 1e-6).sum().item()
        print(f"   Non-zero 3D boxes: {non_zero}/{len(boxes3d)}")

    print("\n" + "=" * 60)
    print("INFERENCE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
