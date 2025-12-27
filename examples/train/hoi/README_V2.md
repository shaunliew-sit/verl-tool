# HOI Detection V2 Training Guide

## Overview

This V2 training setup addresses the issues identified in V1:

1. **Wrong verb predictions** - Model predicting "riding" when ground truth was "holding"
2. **Verbose/repetitive outputs** - Model getting stuck in loops or generating excessive reasoning
3. **Format confusion** - Model not understanding expected output format

## Key Changes in V2

### 1. Improved Reward Function (`hoi_reward_v2.py`)

**Problem**: V1 used word overlap scoring, which gave partial rewards for wrong verbs:
- Ground truth: "holding horse" → Predicted: "riding horse" → Score: 0.5 (wrong!)

**Solution**: V2 uses **verb-first scoring**:
- Verb (first word) MUST match for any positive reward
- Same example now: Score: 0.0 (correct!)

```
Scoring Formula:
- Verb mismatch: 0.0 (no reward for wrong action)
- Verb match + exact phrase: 1.0
- Verb match + partial object: 0.5 + 0.5 * overlap
```

### 2. SFT Before RL

**Problem**: Base model doesn't know:
- Expected output format (short phrases vs. verbose explanations)
- How to use tools appropriately
- When to stop reasoning and provide answer

**Solution**: Add SFT (Supervised Fine-Tuning) stage before RL:
1. Create ideal input-output pairs with correct format
2. Train model to mimic this format
3. Then use RL to optimize correctness

### 3. Training Hyperparameters

| Parameter | V1 | V2 | Reason |
|-----------|----|----|--------|
| Learning Rate | 1e-6 | 5e-7 | More stable learning |
| KL Penalty | 0.0 | 0.001 | Prevent divergence |
| Entropy Coeff | 0.0 | 0.01 | Encourage exploration |
| Total Steps | 100 | 200 | More training |
| Warmup Steps | 10 | 20 | Smoother start |

## Recommended Workflow

### Option A: Full Training (Recommended)

```bash
# Step 1: Prepare SFT data
python examples/data_preprocess/hoi/prepare_sft_data.py

# Step 2: Run SFT training (~3 epochs)
bash examples/train/hoi/train_hoi_sft.sh

# Step 3: Run RL training with SFT checkpoint
MODEL_PATH=./checkpoints/hoi_sft/hoi-sft-qwen3vl-4b/latest \
bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
```

### Option B: RL Only (Faster but less effective)

```bash
# Skip SFT, use V2 reward directly on base model
bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
```

## Files Created

```
examples/train/hoi/
├── train_hoi_qwen3vl_v2.sh    # V2 RL training script
├── train_hoi_sft.sh           # SFT training script
└── README_V2.md               # This file

examples/data_preprocess/hoi/
└── prepare_sft_data.py        # SFT data preparation

verl_tool/workers/reward_manager/
└── hoi_reward_v2.py           # V2 reward with verb-first scoring
```

## Evaluation

After training, use the existing evaluation scripts:

```bash
# Start vLLM server with trained model
bash examples/eval/hoi/start_vllm_server.sh

# Run referring task evaluation
python examples/eval/hoi/eval_hoi_agent.py \
    --endpoint http://localhost:8000/v1 \
    --model hoi-trained \
    --image-path data/hico_20160224_det/images/test2015/HICO_test2015_00000005.jpg \
    --person-bbox "[70, 71, 355, 499]" \
    --object-bbox "[96, 264, 637, 498]" \
    --object-label "bicycle" \
    --max-turns 10 \
    --verbose

# Run grounding task evaluation
python examples/eval/hoi/eval_hoi_agent.py \
    --endpoint http://localhost:8000/v1 \
    --model hoi-trained \
    --image-path data/hico_20160224_det/images/test2015/HICO_test2015_00000002.jpg \
    --action "holding" \
    --object-label "horse" \
    --ground-truth '[{"bbox_2d": [353, 39, 532, 456], "label": "person"}, {"bbox_2d": [272, 141, 615, 956], "label": "horse"}]' \
    --output-viz /tmp/grounding_result.jpg \
    --max-turns 10 \
    --verbose
```

## Expected Improvements

With V2 training:

1. **Verb Accuracy**: Model should correctly identify the action verb
2. **Output Format**: Shorter, more concise responses
3. **Tool Usage**: More appropriate tool calls (not excessive)
4. **Convergence**: Steadier reward curve during training

## Monitoring

Watch for these metrics during training:
- `referring_score`: Should increase steadily
- `verb_match`: New metric - should approach 1.0
- `grounding_score`: Should remain stable/improve
- `response_length`: Should decrease (more concise outputs)

## Troubleshooting

### Tool Server Errors
If you see "meta tensor" errors during training:
```bash
# Ensure tool server has access to GPU 0
CUDA_VISIBLE_DEVICES=0 python -m verl_tool.servers.serve --tool_type hoi_detector
```

### Out of Memory
```bash
# Reduce batch size and increase offloading
N_GPUS=4 BATCH_SIZE=32 DO_OFFLOAD=True bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
```

### SFT Data Issues
```bash
# Verify SFT data was created correctly
wc -l data/hoi/sft_data/train.jsonl
head -1 data/hoi/sft_data/train.jsonl | python -m json.tool
```

