# HOI SFT Dataset Generation and Training

This directory contains scripts for generating high-quality SFT (Supervised Fine-Tuning) datasets for Human-Object Interaction (HOI) detection using a teacher model, following the PixelReasoner methodology.

## Overview

**Key Insight**: We use a teacher model (Qwen3-VL) to generate reasoning traces by prompting it to "think step-by-step" and optionally use tools. The teacher doesn't need native tool-use training - we guide it through structured prompting.

**Approach**:
1. Deploy Qwen3-VL teacher model via vLLM (fast inference)
2. Generate responses for HOI tasks with quality filtering
3. Format data in LLaMA-Factory ShareGPT format
4. Train student model with LoRA using LLaMA-Factory
5. Fine-tune with RL (optional, see parent README)

## Quick Start

### 1. Start vLLM Server (Docker)

First, start the vLLM server using Docker (handles CUDA compatibility):

```bash
# Navigate to vLLM deployment directory
cd vllm-deployment/

# One-time setup: Build and start container
./build_and_run_vllm.sh

# Inside container: Setup Python environment
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto

# Start vLLM server (use screen for persistence)
screen -S vllm
vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --limit-mm-per-prompt.video 0 \
    --async-scheduling

# Detach from screen: Ctrl+A then D
# Reattach later: screen -r vllm
```

The vLLM API will be available at `http://localhost:8000/v1` from your host machine.

### 2. Generate SFT Dataset

Generate a pilot dataset with 10 samples for testing:

```bash
# From verl-tool directory
python examples/data_preprocess/hoi/generate_sft_with_teacher.py \
    --input data/benchmarks_simplified/hico_referring_train_simplified.json \
    --output data/hoi/sft_data/pilot_10.json \
    --api_base http://localhost:8000/v1 \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --num_samples 10 \
    --quality_threshold 0.5
```

**Manual Review**: Check `data/hoi/sft_data/pilot_10.json` to verify:
- Responses make sense for the given queries
- Format is correct (ShareGPT with `messages` and `images`)
- Quality is acceptable

**Scale Up**: After validation, generate more samples:

```bash
# Generate 100 samples for training
python examples/data_preprocess/hoi/generate_sft_with_teacher.py \
    --input data/benchmarks_simplified/hico_referring_train_simplified.json \
    --output data/hoi/sft_data/train.json \
    --api_base http://localhost:8000/v1 \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --num_samples 100 \
    --quality_threshold 0.7

# Generate validation set
python examples/data_preprocess/hoi/generate_sft_with_teacher.py \
    --input data/benchmarks_simplified/hico_referring_train_simplified.json \
    --output data/hoi/sft_data/val.json \
    --api_base http://localhost:8000/v1 \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --num_samples 20 \
    --quality_threshold 0.7
```

### 3. Train with LLaMA-Factory

Train the student model using the generated SFT data:

```bash
bash examples/train/hoi/train_hoi_sft_llamafactory.sh
```

**Monitor Training**:
```bash
tensorboard --logdir checkpoints/hoi_sft_qwen3vl_4b_llamafactory
```

**Merge LoRA Weights** (after training):
```bash
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
ADAPTER_PATH=checkpoints/hoi_sft_qwen3vl_4b_llamafactory
OUTPUT_PATH=checkpoints/hoi_sft_qwen3vl_4b_llamafactory_merged

python -m llamafactory.cli export \
    --model_name_or_path $MODEL_PATH \
    --adapter_name_or_path $ADAPTER_PATH \
    --template qwen3_vl \
    --finetuning_type lora \
    --export_dir $OUTPUT_PATH \
    --export_size 2 \
    --export_legacy_format False
```

## Dataset Format

### ShareGPT Format (LLaMA-Factory)

The generated dataset uses LLaMA-Factory's ShareGPT format for multimodal data:

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant for Human-Object Interaction detection.\n\n# Tools\n\n..."
      },
      {
        "role": "user",
        "content": "<image>Look at the person at region [0.2, 0.3, 0.5, 0.7] and the object at [0.6, 0.4, 0.8, 0.9]. What interaction is happening?\n\nGuidelines: ..."
      },
      {
        "role": "assistant",
        "content": "Looking at the regions specified, I can see a person and a bicycle. The person appears to be positioned on the bicycle in a riding posture. Based on the spatial relationship and body positioning, the interaction is: riding bicycle"
      }
    ],
    "images": [
      "/abs/path/to/image.jpg"
    ]
  }
]
```

**Key Points**:
- Top-level is a JSON array (not JSONL)
- Each example has `messages` and `images` keys
- `content` is a simple string (use `<image>` placeholder)
- `images` is an array of absolute paths
- No `loss_weight` or complex content structures

## Script Reference

### `generate_sft_with_teacher.py`

Generate SFT dataset using teacher model via vLLM API.

**Key Arguments**:
- `--input`: Input JSON file with HOI benchmark samples
- `--output`: Output JSON file for SFT data (ShareGPT format)
- `--api_base`: vLLM API endpoint (default: `http://localhost:8000/v1`)
- `--model_name`: Teacher model name (e.g., `Qwen/Qwen3-VL-4B-Instruct`)
- `--num_samples`: Number of samples to generate
- `--quality_threshold`: Minimum quality score to keep (0.0-1.0)
- `--max_per_action`: Maximum samples per action category (default: 20)

**Quality Evaluation**:
- For referring tasks: Checks if predicted action matches ground truth
- For grounding tasks: Checks if response contains bounding box format
- Only responses above threshold are kept

**Sample Selection**:
- Diverse action categories (avoids over-representation)
- Random sampling with category balancing
- Challenging cases preferred

### `train_hoi_sft_llamafactory.sh`

Train student model with LLaMA-Factory using LoRA.

**Configuration** (edit script to customize):
- `MODEL_PATH`: Base model (default: `Qwen/Qwen3-VL-4B-Instruct`)
- `LEARNING_RATE`: Learning rate (default: `2e-5`)
- `NUM_EPOCHS`: Training epochs (default: `3`)
- `BATCH_SIZE`: Per-device batch size (default: `4`)
- `GRAD_ACCUM_STEPS`: Gradient accumulation (default: `8`)
- `LORA_RANK`: LoRA rank (default: `64`)
- `LORA_ALPHA`: LoRA alpha (default: `128`)

## Production Workflow

### Full Pipeline (500-1000 High-Quality Examples)

**Step 1: Deploy Better Teacher Model**

For production, use a larger teacher model:

```bash
# Inside vLLM container
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --limit-mm-per-prompt.video 0 \
    --async-scheduling
```

**Step 2: Generate Training Data**

```bash
# Generate referring task data (250 samples)
python examples/data_preprocess/hoi/generate_sft_with_teacher.py \
    --input data/benchmarks_simplified/hico_referring_train_simplified.json \
    --output data/hoi/sft_data/train_referring.json \
    --api_base http://localhost:8000/v1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --num_samples 250 \
    --quality_threshold 0.7

# Generate grounding task data (250 samples)
python examples/data_preprocess/hoi/generate_sft_with_teacher.py \
    --input data/benchmarks_simplified/hico_ground_train_simplified.json \
    --output data/hoi/sft_data/train_grounding.json \
    --api_base http://localhost:8000/v1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --num_samples 250 \
    --quality_threshold 0.7

# Merge datasets
python -c "
import json
with open('data/hoi/sft_data/train_referring.json') as f:
    referring = json.load(f)
with open('data/hoi/sft_data/train_grounding.json') as f:
    grounding = json.load(f)
merged = referring + grounding
with open('data/hoi/sft_data/train.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged {len(merged)} examples')
"
```

**Step 3: Generate Validation Data**

```bash
# Similar to training, but with fewer samples (50 total)
python examples/data_preprocess/hoi/generate_sft_with_teacher.py \
    --input data/benchmarks_simplified/hico_referring_train_simplified.json \
    --output data/hoi/sft_data/val.json \
    --api_base http://localhost:8000/v1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --num_samples 50 \
    --quality_threshold 0.7
```

**Step 4: Train Student Model**

```bash
bash examples/train/hoi/train_hoi_sft_llamafactory.sh
```

**Step 5: Evaluate SFT Model**

```bash
# TODO: Add evaluation script for HOI benchmark
# Compare SFT model vs base model on test set
```

## Troubleshooting

### vLLM Server Issues

**Issue**: CUDA PTX compatibility errors
- **Solution**: Use Docker method (handles CUDA 12.9 compatibility)

**Issue**: Out of memory during generation
- **Solution**: Lower `--tensor-parallel-size` or use smaller model (4B)

**Issue**: API connection refused
- **Solution**: Check if vLLM server is running (`screen -r vllm`)

### Dataset Generation Issues

**Issue**: No examples generated (success rate 0%)
- **Solution**: Lower `--quality_threshold` (e.g., 0.3 for testing)
- **Check**: Verify image paths exist and are accessible

**Issue**: Quality scores too low
- **Solution**: Use better teacher model (8B or 235B instead of 4B)
- **Check**: Review example responses manually to debug evaluation logic

**Issue**: API call timeout
- **Solution**: Increase `max_tokens` or reduce `num_samples` per batch

### Training Issues

**Issue**: Dataset not found
- **Solution**: Ensure `train.json` exists in `data/hoi/sft_data/`
- **Check**: Run generation script first

**Issue**: Out of memory during training
- **Solution**: Reduce `BATCH_SIZE` or increase `GRAD_ACCUM_STEPS`
- **Try**: Enable CPU offloading in LLaMA-Factory config

## References

- **PixelReasoner**: Teacher-in-the-Loop methodology for vision-language tool use
- **LLaMA-Factory**: Efficient fine-tuning framework with ShareGPT support
- **vLLM**: Fast inference engine with OpenAI-compatible API
- **verl**: RL framework for tool-calling agents (next stage after SFT)

## Next Steps

After SFT training completes:

1. **Evaluate SFT Model**: Test on HOI benchmark to verify improvement over base model
2. **RL Fine-Tuning**: Use SFT model as initialization for RL training (see parent README)
3. **Iteration**: If SFT model quality is low, refine dataset and retrain

**Target Metrics** (after SFT):
- Referring task: 50-60% accuracy (vs 20-30% for base model)
- Grounding task: 30-40% mAP (vs 10-20% for base model)

If metrics are significantly lower, consider:
- Generating more training data (500-1000 examples)
- Using better teacher model (235B)
- Adjusting quality threshold or evaluation criteria
