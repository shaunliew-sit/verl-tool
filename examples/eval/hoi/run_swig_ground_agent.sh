#!/bin/bash
# Run SWIG Grounding Evaluation with HOI Agent
#
# Prerequisites:
#   1. Start vLLM server first:
#      CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/eval/hoi/start_vllm_server.sh \
#          checkpoints/.../actor/huggingface 8000 hoi-trained 4
#
# Usage:
#   bash examples/eval/hoi/run_swig_ground_agent.sh
#
# Environment variables:
#   ENDPOINT      - vLLM endpoint (default: http://localhost:8000/v1)
#   MODEL         - Model name (default: hoi-trained)
#   MAX_IMAGES    - Max images to evaluate (default: all)
#   CONCURRENCY   - Async concurrency (default: 8)
#   VERBOSE       - Enable verbose output (default: false)
#   SAVE_VIZ      - Save visualization images (default: false, auto-enabled with VERBOSE)
#   UNIQUE_RUN    - Append timestamp to output dir (default: true)
#   WANDB         - Enable W&B logging (default: false)
#   OUTPUT_DIR    - Output directory (default: results/swig_ground_agent)

set -e

# Configuration
ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
MODEL="${MODEL:-hoi-trained}"
MAX_IMAGES="${MAX_IMAGES:-}"
CONCURRENCY="${CONCURRENCY:-8}"
VERBOSE="${VERBOSE:-false}"
SAVE_VIZ="${SAVE_VIZ:-false}"
UNIQUE_RUN="${UNIQUE_RUN:-true}"
WANDB="${WANDB:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-results/swig_ground_agent}"

# Dataset paths
ANN_FILE="data/benchmarks_simplified/swig_ground_test_simplified.json"
IMG_PREFIX="data/swig_hoi/images_512"

echo "=============================================="
echo "SWIG Grounding Evaluation"
echo "=============================================="
echo "Endpoint:    $ENDPOINT"
echo "Model:       $MODEL"
echo "Annotation:  $ANN_FILE"
echo "Images:      $IMG_PREFIX"
echo "Concurrency: $CONCURRENCY"
echo "Output:      $OUTPUT_DIR"
echo "=============================================="

# Build command
CMD="python examples/eval/hoi/eval_hoi_agent.py \
    --task grounding \
    --dataset swig \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --endpoint $ENDPOINT \
    --model $MODEL \
    --concurrency $CONCURRENCY \
    --output-dir $OUTPUT_DIR \
    --save-thinking"

# Add optional flags
if [ -n "$MAX_IMAGES" ]; then
    CMD="$CMD --max-images $MAX_IMAGES"
fi

if [ "$VERBOSE" = "true" ]; then
    CMD="$CMD --verbose"
fi

if [ "$SAVE_VIZ" = "true" ]; then
    CMD="$CMD --save-viz"
fi

if [ "$UNIQUE_RUN" = "false" ]; then
    CMD="$CMD --no-unique-run"
fi

if [ "$WANDB" = "true" ]; then
    CMD="$CMD --wandb"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to output directory"
echo "=============================================="

