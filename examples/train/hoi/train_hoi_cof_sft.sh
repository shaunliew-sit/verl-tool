#!/bin/bash
# HOI Chain-of-Focus SFT Training with LLaMA-Factory
# Model: Qwen3-VL-8B-Instruct with LoRA
# Dataset: hoi_cof_sft (5k samples)
# Hardware: 8 GPUs

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Log folder setup
LOG_FOLDER="$PROJECT_ROOT/logs"
mkdir -p "$LOG_FOLDER"

# Config file path
CONFIG_FILE="$SCRIPT_DIR/hoi_cof_sft_lora.yaml"

# GPU Configuration
NUM_GPUS=8
MASTER_PORT=29500

# Environment setup
export DISABLE_VERSION_CHECK=1
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Verify dataset exists
DATASET_DIR="$PROJECT_ROOT/data/hoi_cof_sft"
if [ ! -f "$DATASET_DIR/hoi_cof_sft_data.json" ]; then
    echo "Error: Dataset not found at $DATASET_DIR/hoi_cof_sft_data.json"
    echo "Please ensure the HOI SFT dataset is generated."
    exit 1
fi

if [ ! -f "$DATASET_DIR/dataset_info.json" ]; then
    echo "Error: dataset_info.json not found at $DATASET_DIR"
    exit 1
fi

echo "============================================"
echo "HOI Chain-of-Focus SFT Training (8 GPUs)"
echo "============================================"
echo "Config: $CONFIG_FILE"
echo "Dataset: $DATASET_DIR"
echo "Working Dir: $PROJECT_ROOT"
echo "GPUs: $NUM_GPUS"
echo "Log File: $LOG_FOLDER/train-hoi-cof-sft-8b-lora.log"
echo "Start Time: $(date)"
echo "============================================"
echo ""

# Change to project root so relative paths work
cd "$PROJECT_ROOT"

# Multi-GPU training with torchrun
echo "Starting 8-GPU training... (logs saved to $LOG_FOLDER/train-hoi-cof-sft-8b-lora.log)"
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    -m llamafactory.cli train "$CONFIG_FILE" > "$LOG_FOLDER/train-hoi-cof-sft-8b-lora.log" 2>&1 &

wait

echo ""
echo "============================================"
echo "Finish Training!!! [$(date)]"
echo "============================================"
echo "Model saved to: saves/qwen3-vl-8b/lora/hoi_cof_sft"
echo "Training log: $LOG_FOLDER/train-hoi-cof-sft-8b-lora.log"
echo ""
echo "To merge LoRA weights with base model:"
echo "  python -m llamafactory.cli export \\"
echo "    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \\"
echo "    --adapter_name_or_path saves/qwen3-vl-8b/lora/hoi_cof_sft \\"
echo "    --template qwen3_vl_nothink \\"
echo "    --finetuning_type lora \\"
echo "    --export_dir saves/qwen3-vl-8b/merged/hoi_cof_sft \\"
echo "    --export_size 2 \\"
echo "    --export_legacy_format False"
echo ""
echo "To view wandb logs:"
echo "  Visit https://wandb.ai"
