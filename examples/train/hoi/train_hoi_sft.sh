#!/bin/bash
# HOI Detection SFT (Supervised Fine-Tuning) Training Script
#
# PURPOSE:
# Train the base model on ideal input-output pairs BEFORE running RL.
# This teaches the model:
#   1. The expected output format (short action phrases, bounding box JSON)
#   2. How to use tools appropriately
#   3. How to structure reasoning for HOI tasks
#
# RECOMMENDED WORKFLOW:
#   1. Prepare SFT data: python examples/data_preprocess/hoi/prepare_sft_data.py
#   2. Run SFT training: bash examples/train/hoi/train_hoi_sft.sh
#   3. Run RL training: bash examples/train/hoi/train_hoi_qwen3vl_v2.sh (using SFT checkpoint)
#
# Usage:
#   bash examples/train/hoi/train_hoi_sft.sh
#
# GPU Memory: ~40GB per GPU for Qwen3-VL-4B with gradient checkpointing

set -x

# Configuration
nproc_per_node=${N_GPUS:-4}
model_name=Qwen/Qwen3-VL-4B-Instruct
save_path=./checkpoints/hoi_sft
train_data=$(pwd)/data/hoi/sft_data/train.jsonl
val_data=$(pwd)/data/hoi/sft_data/eval.jsonl

# Training parameters
max_length=4096  # Max sequence length
batch_size=64
micro_batch_size=2
learning_rate=2e-5
total_epochs=3

echo "===== HOI SFT Training ====="
echo "Model: $model_name"
echo "Training data: $train_data"
echo "GPUs: $nproc_per_node"
echo "Batch size: $batch_size"
echo "Max length: $max_length"
echo "============================"

# Check if SFT data exists
if [ ! -f "$train_data" ]; then
    echo "ERROR: SFT training data not found at $train_data"
    echo "Please run: python examples/data_preprocess/hoi/prepare_sft_data.py"
    exit 1
fi

# Run SFT training using multi-turn format
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.multiturn.enable=True \
    data.multiturn.messages_key=messages \
    data.micro_batch_size_per_gpu=$micro_batch_size \
    data.max_length=$max_length \
    data.train_batch_size=$batch_size \
    data.truncation=right \
    use_remove_padding=True \
    model.partial_pretrain=$model_name \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    optim.lr=$learning_rate \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=hoi_sft \
    trainer.experiment_name=hoi-sft-qwen3vl-4b \
    trainer.total_epochs=$total_epochs \
    trainer.save_freq=500 \
    trainer.test_freq=100 \
    trainer.logger=['console','wandb']

echo ""
echo "===== SFT Training Complete ====="
echo "Checkpoint saved to: $save_path"
echo ""
echo "Next step: Run RL training with the SFT checkpoint:"
echo "  MODEL_PATH=$save_path/hoi-sft-qwen3vl-4b/latest \\"
echo "  bash examples/train/hoi/train_hoi_qwen3vl_v2.sh"

