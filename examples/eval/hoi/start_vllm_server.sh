#!/bin/bash
# Start vLLM server with the trained HOI model
#
# Usage:
#   bash examples/eval/hoi/start_vllm_server.sh [checkpoint_path] [port]
#
# Examples:
#   # Use latest checkpoint (default)
#   bash examples/eval/hoi/start_vllm_server.sh
#
#   # Use specific checkpoint
#   bash examples/eval/hoi/start_vllm_server.sh checkpoints/.../global_step_100/actor/huggingface 8000

set -e

# Default values
DEFAULT_CHECKPOINT="checkpoints/hoi_reward/hoi_reward-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b64-t1.0-lr1e-6-hoi-detection/global_step_100/actor/huggingface"
DEFAULT_PORT=8000
DEFAULT_MODEL_NAME="hoi-trained"

# Parse arguments
CHECKPOINT_PATH="${1:-$DEFAULT_CHECKPOINT}"
PORT="${2:-$DEFAULT_PORT}"
MODEL_NAME="${3:-$DEFAULT_MODEL_NAME}"

echo "=============================================="
echo "Starting vLLM Server for HOI Evaluation"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Port: $PORT"
echo "Model name: $MODEL_NAME"
echo "=============================================="

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    find checkpoints -name "huggingface" -type d 2>/dev/null || echo "  No checkpoints found"
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
echo "  2. Run evaluation: python examples/eval/hoi/eval_hoi_agent.py --endpoint http://localhost:$PORT/v1 --model $MODEL_NAME --interactive --image-path <image>"
echo ""

# Start vLLM server with tool calling enabled
python -m vllm.entrypoints.openai.api_server \
    --model "$CHECKPOINT_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

