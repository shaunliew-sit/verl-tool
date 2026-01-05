#!/bin/bash
# Start vLLM server with the trained HOI model (Multi-GPU support)
#
# Usage:
#   bash examples/eval/hoi/start_vllm_server.sh [checkpoint_path] [port] [model_name] [tensor_parallel_size]
#
# Examples:
#   # Single GPU (default)
#   bash examples/eval/hoi/start_vllm_server.sh
#
#   # 4 GPU tensor parallelism
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/eval/hoi/start_vllm_server.sh \
#       checkpoints/.../global_step_100/actor/huggingface 8000 hoi-trained 4
#
#   # 8 GPU tensor parallelism
#   bash examples/eval/hoi/start_vllm_server.sh \
#       checkpoints/.../global_step_100/actor/huggingface 8000 hoi-trained 8

set -e

# Default values
DEFAULT_CHECKPOINT="checkpoints/hoi_reward_v2/hoi_reward_v2-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b128-t1.0-lr5e-7-hoi-detection-v2/global_step_100/actor/huggingface"
DEFAULT_PORT=8000
DEFAULT_MODEL_NAME="hoi-trained"
DEFAULT_TP_SIZE=1

# Parse arguments
CHECKPOINT_PATH="${1:-$DEFAULT_CHECKPOINT}"
PORT="${2:-$DEFAULT_PORT}"
MODEL_NAME="${3:-$DEFAULT_MODEL_NAME}"
TP_SIZE="${4:-$DEFAULT_TP_SIZE}"

# Get GPU memory utilization (lower for multi-GPU)
if [ "$TP_SIZE" -gt 1 ]; then
    GPU_MEM_UTIL=0.85
else
    GPU_MEM_UTIL=0.8
fi

echo "=============================================="
echo "Starting vLLM Server for HOI Evaluation"
echo "=============================================="
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Port:            $PORT"
echo "Model name:      $MODEL_NAME"
echo "Tensor Parallel: $TP_SIZE GPUs"
echo "GPU Memory Util: $GPU_MEM_UTIL"
echo "=============================================="

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    find checkpoints -name "huggingface" -type d 2>/dev/null | head -20 || echo "  No checkpoints found"
    exit 1
fi

# Check for config.json
if [ ! -f "$CHECKPOINT_PATH/config.json" ]; then
    echo "ERROR: config.json not found in checkpoint directory"
    exit 1
fi

echo ""
echo "Starting vLLM server..."
echo "This may take a few minutes to load the model."
echo ""
echo "Once started, you can:"
echo "  1. Test with: curl http://localhost:$PORT/v1/models"
echo "  2. Run grounding evaluation:"
echo "     python examples/eval/hoi/eval_hoi_agent.py \\"
echo "         --task grounding --dataset hico \\"
echo "         --ann-file data/benchmarks_simplified/hico_ground_test_simplified.json \\"
echo "         --img-prefix data/hico_20160224_det/images/test2015 \\"
echo "         --endpoint http://localhost:$PORT/v1 --model $MODEL_NAME \\"
echo "         --concurrency 8 --save-thinking --output-dir results/hico_ground"
echo ""
echo "  3. Run referring evaluation:"
echo "     python examples/eval/hoi/eval_hoi_agent.py \\"
echo "         --task referring --dataset hico \\"
echo "         --ann-file data/benchmarks_simplified/hico_action_referring_test_simplified.json \\"
echo "         --img-prefix data/hico_20160224_det/images/test2015 \\"
echo "         --endpoint http://localhost:$PORT/v1 --model $MODEL_NAME \\"
echo "         --concurrency 8 --save-thinking --output-dir results/hico_referring"
echo ""

# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model \"$CHECKPOINT_PATH\" \
    --served-model-name \"$MODEL_NAME\" \
    --port $PORT \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --enable-auto-tool-choice \
    --tool-call-parser hermes"

# Add tensor parallelism if using multiple GPUs
if [ "$TP_SIZE" -gt 1 ]; then
    VLLM_CMD="$VLLM_CMD --tensor-parallel-size $TP_SIZE"
fi

# Execute
echo "Running: $VLLM_CMD"
echo ""
eval $VLLM_CMD
