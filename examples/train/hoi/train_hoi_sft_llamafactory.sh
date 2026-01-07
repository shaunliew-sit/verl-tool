#!/bin/bash
# SFT training with LLaMA-Factory for HOI detection
# This script trains a Qwen3-VL model on HOI SFT dataset using LoRA

set -e

# Activate virtual environment if needed
# source .venv/bin/activate

# Paths
TRAIN_DATA=data/hoi/sft_data/train.json
VAL_DATA=data/hoi/sft_data/val.json
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
OUTPUT_DIR=checkpoints/hoi_sft_qwen3vl_4b_llamafactory

# Dataset directory (contains dataset_info.json)
DATASET_DIR=data/hoi/sft_data

# Training hyperparameters
LEARNING_RATE=2e-5
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM_STEPS=8

# LoRA configuration
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.1

# Verify data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Please run generate_sft_with_teacher.py first to generate the SFT dataset"
    echo ""
    echo "Example command:"
    echo "  python examples/data_preprocess/hoi/generate_sft_with_teacher.py \\"
    echo "    --input data/benchmarks_simplified/hico_referring_train_simplified.json \\"
    echo "    --output data/hoi/sft_data/train.json \\"
    echo "    --api_base http://localhost:8000/v1 \\"
    echo "    --model_name Qwen/Qwen3-VL-4B-Instruct \\"
    echo "    --num_samples 500 \\"
    echo "    --quality_threshold 0.7"
    exit 1
fi

if [ ! -f "$DATASET_DIR/dataset_info.json" ]; then
    echo "Error: dataset_info.json not found at $DATASET_DIR"
    echo "Please ensure the dataset configuration file exists"
    exit 1
fi

echo "============================================"
echo "HOI SFT Training with LLaMA-Factory"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Train Data: $TRAIN_DATA"
echo "Val Data: $VAL_DATA"
echo "Output Dir: $OUTPUT_DIR"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE (grad accum: $GRAD_ACCUM_STEPS)"
echo "LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "============================================"
echo ""

# Training with LLaMA-Factory
python -m llamafactory.cli train \
    --stage sft \
    --model_name_or_path $MODEL_PATH \
    --do_train True \
    --do_eval True \
    --train_dataset hoi_sft_train \
    --eval_dataset hoi_sft_val \
    --dataset_dir $DATASET_DIR \
    --template qwen3_vl \
    --finetuning_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target all \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --report_to tensorboard \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss

echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To merge LoRA weights with base model:"
echo "  python -m llamafactory.cli export \\"
echo "    --model_name_or_path $MODEL_PATH \\"
echo "    --adapter_name_or_path $OUTPUT_DIR \\"
echo "    --template qwen3_vl \\"
echo "    --finetuning_type lora \\"
echo "    --export_dir ${OUTPUT_DIR}_merged \\"
echo "    --export_size 2 \\"
echo "    --export_legacy_format False"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir $OUTPUT_DIR"
