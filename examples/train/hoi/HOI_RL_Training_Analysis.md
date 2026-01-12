# HOI RL Training Setup Analysis

## Overview

The HOI (Human-Object Interaction) RL training setup follows a similar architecture to Pixel Reasoner, using tool-assisted reinforcement learning for visual reasoning tasks. The system trains models to perform two tasks: **referring** (predict action given person-object boxes) and **grounding** (locate person-object pairs given action).

---

## 1. RL Data Preparation

### Data Sources
- **HICO-DET**: Human-Object Interaction dataset
- **SWIG-HOI**: Scene-Wide Interaction Graph dataset

### Preparation Script
```bash
python examples/data_preprocess/hoi/prepare_hoi.py \
    --local_dir data/hoi/train_data \
    --seed 42
```

### Data Format
- **Input**: Simplified JSON files from `data/benchmarks_simplified/`
- **Output**: Parquet files (`train.parquet`, `val.parquet`) compatible with verl-tool

### Data Structure
Each sample contains:
- `prompt`: Multi-turn conversation with system prompt + user query
- `images`: Image paths (absolute paths)
- `reward_model`: Ground truth and task type
  - `ground_truth`: Action phrase (referring) or bounding boxes (grounding)
  - `task_type`: `"referring"` or `"grounding"`
- `data_source`: Dataset identifier (`hico_ground`, `hico_referring`, etc.)

### Coordinate System
- All bounding boxes normalized to **1000×1000** format (Qwen3-VL compatible)
- Conversion: `coord_1000 = int(coord_pixel / (dim - 1) * 1000)`

---

## 2. Tool Design

### Tool Implementation
**File**: `verl_tool/servers/tools/hoi_detector.py`

### Available Tools

#### 1. `zoom_in`
- **Purpose**: Crop and zoom into a specific image region
- **Arguments**:
  - `bbox_2d`: `[x1, y1, x2, y2]` in 1000×1000 normalized format
  - `target_image`: Image index (typically `1`)
- **Behavior**: Crops image region, updates environment state

#### 2. `zoom_out`
- **Purpose**: Reset to original full image view
- **Arguments**:
  - `target_image`: Image index (typically `1`)
- **Behavior**: Restores original image in environment

#### 3. `detect_objects`
- **Purpose**: Object detection using Grounding DINO
- **Arguments**:
  - `class_names`: Query string (e.g., `"person . bicycle"`)
  - `target_image`: Image index (typically `1`)
  - `confidence_threshold`: Optional (default: 0.25)
- **Backend**: Uses `IDEA-Research/grounding-dino-tiny` model
- **Output**: List of detections with `{label, bbox, confidence}`

### Tool Server
- **Type**: `hoi_detector`
- **Start Command**:
```bash
python -m verl_tool.servers.serve --tool_type hoi_detector --workers_per_tool 4
```
- **Architecture**: Similar to Pixel Reasoner's tool server
  - Maintains per-trajectory environment state
  - Handles image cropping and object detection
  - Returns observations as text descriptions

---

## 3. Reward Design

### Reward Manager
**File**: `verl_tool/workers/reward_manager/hoi_reward_v2.py`

### Key Improvement (V2)
**Verb-First Scoring**: Action verb MUST match for any positive reward (fixes "riding" vs "holding" confusion from V1).

### Reward Functions

#### Referring Task (`referring_score_v2`)
**Scoring Formula**:
```
- Verb mismatch: 0.0 (no reward for wrong action)
- Verb match + exact phrase match: 1.0
- Verb match + partial object match: 0.5 + 0.5 * object_overlap
```

**Process**:
1. Extract action phrase from response (multiple strategies)
2. Normalize verbs (handles -ing forms, synonyms)
3. Compare first word (verb) - **strict requirement**
4. If verb matches, compute object overlap

**Example**:
- GT: `"holding horse"` → Pred: `"riding horse"` → **Score: 0.0** ✓
- GT: `"holding horse"` → Pred: `"holding pony"` → **Score: 0.5 + 0.5*overlap**

#### Grounding Task (`grounding_score`)
**Scoring Formula**:
```
- Binary reward: 1.0 if all GT boxes matched (IoU ≥ 0.5), else 0.0
```

**Process**:
1. Extract bounding boxes from JSON response
2. Compute IoU between predicted and ground truth boxes
3. Match each GT box to best prediction (IoU ≥ 0.5)
4. Reward = 1.0 if all GT boxes matched

### Action Phrase Extraction
Multiple strategies (in priority order):
1. `"Final Answer:"` pattern
2. `"ACTION:"` pattern
3. `"action phrase is:"` pattern
4. Last short line (2-4 words, verb + noun)
5. Verb + object pattern in last 200 chars

---

## 4. Training Configuration

### Key Hyperparameters (V2 Improvements)

| Parameter | V1 | V2 | Reason |
|-----------|----|----|--------|
| Learning Rate | 1e-6 | **5e-7** | More stable learning |
| KL Penalty | 0.0 | **0.001** | Prevent divergence |
| Entropy Coeff | 0.0 | **0.01** | Encourage exploration |
| Total Steps | 100 | **200** | More training |
| Warmup Steps | 10 | **20** | Smoother start |

### Training Strategy
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Strategy**: FSDP2 (Fully Sharded Data Parallel)
- **Samples per prompt**: `n=8` (for group normalization)
- **Batch size**: 128 (default, adjustable via `BATCH_SIZE`)

### Sequence Lengths
- `max_prompt_length`: 16384 (adjustable: `MAX_PROMPT_LEN`)
- `max_response_length`: 8192 (adjustable: `MAX_RESPONSE_LEN`)
- `max_action_length`: 2048
- `max_obs_length`: 4096

### Agent Configuration
- `enable_agent`: `True` (tool use enabled)
- `max_turns`: 3 (maximum tool interaction turns)
- `action_stop_tokens`: `'</tool_call>'`
- `enable_mtrl`: `True` (multi-turn RL)

### Memory Optimization
- `do_offload`: `False` (set `True` for 4 GPUs)
- `gpu_memory_utilization`: 0.7
- `tensor_model_parallel_size`: 2
- `max_num_batched_tokens`: 10000

---

## 5. Training Command

### Full Training (Recommended)
```bash
# Step 1: Prepare RL data
python examples/data_preprocess/hoi/prepare_hoi.py \
    --local_dir data/hoi/train_data

# Step 2: (Optional) SFT pre-training
bash examples/train/hoi/train_hoi_sft.sh

# Step 3: RL training
MODEL_PATH=./checkpoints/hoi_sft/hoi-sft-qwen3vl-4b/latest \
bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
```

### RL Only (Faster)
```bash
bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
```

### 4 GPU Configuration (H100 80GB)
```bash
N_GPUS=4 BATCH_SIZE=64 TP_SIZE=2 GPU_MEM_UTIL=0.7 DO_OFFLOAD=True \
MAX_PROMPT_LEN=8192 MAX_RESPONSE_LEN=4096 MAX_BATCHED_TOKENS=8000 \
bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
```

### Training Script Key Components
- **Reward Manager**: `hoi_reward_v2` (verb-first scoring)
- **Model**: `Qwen/Qwen3-VL-4B-Instruct` (or SFT checkpoint)
- **Tool Server**: Auto-started on random port (30000-31000)
- **Checkpointing**: Every 10 steps, saves model + optimizer + HF format

---

## Comparison with Pixel Reasoner

### Similarities
1. **Tool Architecture**: Both use `zoom_in`, `zoom_out` for visual reasoning
2. **Reward Design**: Binary/exact match rewards with task-specific scoring
3. **Multi-turn RL**: Both support multi-turn tool interactions (`enable_mtrl=True`)
4. **Data Format**: Parquet-based, similar prompt structure

### Differences
1. **Tools**: HOI adds `detect_objects` (Grounding DINO) for object detection
2. **Reward**: HOI uses verb-first scoring (referring) vs. mathematical verification (Pixel Reasoner)
3. **Tasks**: HOI focuses on action prediction and grounding vs. math problem solving
4. **Coordinate System**: HOI uses 1000×1000 normalization vs. pixel coordinates

---

## Files Created

```
verl_tool/
├── servers/tools/
│   └── hoi_detector.py              # Tool implementation
└── workers/reward_manager/
    └── hoi_reward_v2.py             # V2 reward function

examples/
├── data_preprocess/hoi/
│   └── prepare_hoi.py               # RL data preparation
└── train/hoi/
    ├── train_hoi_qwen3vl_v2.sh      # V2 RL training script
    └── README_V2.md                  # Training guide
```
