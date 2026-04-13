#!/bin/bash
# ===========================================================================
# RFDet3D Training Script for RunPod
#
# Usage:
#   1. Launch a RunPod instance (A100 40GB recommended)
#   2. Clone the repo:
#        git clone --recurse-submodules git@github.com:Barath19/RFDet3D.git
#        cd RFDet3D
#   3. Set your wandb key:
#        export WANDB_API_KEY="your-key-here"
#   4. Run:
#        bash run_train.sh
# ===========================================================================

set -euo pipefail

# ---- Config (edit these) ----
RFDETR_VARIANT="base"          # nano|small|base|medium|large
NUM_CLASSES=17                 # ARKitScenes 17 classes (change to 80 for COCO)
PHASE=1                        # 1=freeze RF-DETR, 2=fine-tune all
EPOCHS=24
BATCH_SIZE=4                   # Adjust for GPU memory (A100: 8, A6000: 4, 3090: 2)
LR=1e-4
TARGET_SIZE=560                # RF-DETR Base=560, Nano=384

# Data (set to your dataset path after download)
DATA_ROOT="data/train"
VAL_DATA_ROOT="data/val"

# Optional: path to existing WildDet3D checkpoint for 3D head weight transfer
WILDDET3D_CKPT=""              # e.g., "ckpt/wilddet3d_alldata_all_prompt_v1.0.pt"

# wandb
WANDB_PROJECT="rfdet3d"
WANDB_NAME="rfdet3d-${RFDETR_VARIANT}-phase${PHASE}"

OUTPUT_DIR="outputs/${WANDB_NAME}"

# ---- Step 1: Environment setup ----
echo "=========================================="
echo "Step 1: Setting up environment"
echo "=========================================="

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python and dependencies
uv python install 3.11 2>/dev/null || true
uv sync

# Verify
uv run python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')"
uv run python -c "import rfdetr; print('rfdetr OK')"
uv run python -c "import wandb; print('wandb OK')"

# ---- Step 2: Download WildDet3D-Data (if not present) ----
echo ""
echo "=========================================="
echo "Step 2: Data setup"
echo "=========================================="

if [ ! -d "$DATA_ROOT" ]; then
    echo "No data found at ${DATA_ROOT}. Downloading ARKitScenes..."
    echo ""

    # Download ARKitScenes 3DOD (Validation: ~60GB, Training: ~560GB)
    # Change --split and --max_scenes to control download size
    uv run python scripts/prepare_arkitscenes.py \
        --split Training \
        --output_dir "${DATA_ROOT}" \
        --download_dir data/arkitscenes_raw \
        --frame_step 10 \
        --max_frames_per_scene 50

    echo "ARKitScenes prepared at ${DATA_ROOT}/"
    echo ""

    # To test with just a few scenes first, use:
    #   --split Validation --max_scenes 10
fi

# ---- Step 3: Download model checkpoint (optional) ----
echo ""
echo "=========================================="
echo "Step 3: Model weights"
echo "=========================================="

if [ -n "$WILDDET3D_CKPT" ] && [ ! -f "$WILDDET3D_CKPT" ]; then
    echo "Downloading WildDet3D checkpoint for 3D head weights..."
    mkdir -p ckpt
    pip install huggingface_hub -q
    huggingface-cli download allenai/WildDet3D wilddet3d_alldata_all_prompt_v1.0.pt --local-dir ckpt/
    WILDDET3D_CKPT="ckpt/wilddet3d_alldata_all_prompt_v1.0.pt"
fi

echo "RF-DETR ${RFDETR_VARIANT} weights will be auto-downloaded on first run."

# ---- Step 4: Train ----
echo ""
echo "=========================================="
echo "Step 4: Training"
echo "=========================================="
echo "Variant:     ${RFDETR_VARIANT}"
echo "Phase:       ${PHASE}"
echo "Epochs:      ${EPOCHS}"
echo "Batch size:  ${BATCH_SIZE}"
echo "LR:          ${LR}"
echo "Output:      ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

CKPT_FLAG=""
if [ -n "$WILDDET3D_CKPT" ]; then
    CKPT_FLAG="--wilddet3d_ckpt ${WILDDET3D_CKPT}"
fi

uv run python train.py \
    --rfdetr_variant "${RFDETR_VARIANT}" \
    --num_classes "${NUM_CLASSES}" \
    --data_root "${DATA_ROOT}" \
    --target_size "${TARGET_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --phase "${PHASE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "${WANDB_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 4 \
    --log_interval 10 \
    --save_interval 4 \
    ${CKPT_FLAG}

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/"
echo "View logs at: https://wandb.ai"
echo "=========================================="
