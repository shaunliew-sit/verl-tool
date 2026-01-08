# HOI Data Preparation for SFT and RL Training

This directory contains scripts for generating HOI (Human-Object Interaction) datasets for both **SFT (Supervised Fine-Tuning)** and **RL (Reinforcement Learning)** training.

## Overview

The Chain-of-Focus (CoF) approach teaches models to:
1. **Think** before acting (`<think>...</think>` tags)
2. **Use tools** when needed (`<tool_call>...</tool_call>` tags)
3. **Provide structured answers** (`<answer>...</answer>` tags)

### SFT/RL Alignment

Both SFT and RL pipelines are **fully aligned** with Chain-of-Focus methodology:
- **Single tool**: Only `zoom_in` (following CoF paper design)
- **Same system prompt**: Identical tool definitions in both pipelines
- **Same user prompt format**: Consistent task instructions
- **Same reasoning format**: `<think>`, `<tool_call>`, `<answer>` tags

---

# SFT Data Preparation

## Scripts

### `generate_hoi_cof_sft.py` - Main SFT Data Generator

Generates Chain-of-Focus style SFT dataset with:
- Multi-turn conversations (zoom in when needed)
- Single-turn conversations (direct answer when clear)
- Actual cropped/zoomed images for tool calls

**Usage:**
```bash
python examples/data_preprocess/hoi/generate_hoi_cof_sft.py \
    --output_dir data/hoi_cof_sft \
    --max_samples 5000 \
    --zoom_threshold 0.15 \
    --include_hico \
    --include_swig
```

**Output Format:**
```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant...<tools>...</tools>"},
        {"role": "user", "content": "<image> Question: What action..."},
        {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},
        {"role": "user", "content": "<image>\nThink in the mind first..."},
        {"role": "assistant", "content": "<think>...</think>\n<answer>...</answer>"}
    ],
    "images": ["0/0.jpg", "0/1.jpg"]
}
```

### `prepare_hoi.py` - RL Data Preparation

Converts benchmark data to parquet format for RL training. See [RL Data Preparation](#rl-data-preparation) section below for details.

### `prepare_sft_data.py` - Alternative SFT Data

Alternative SFT preparation (less recommended, use `generate_hoi_cof_sft.py` instead).

## SFT Dataset Statistics

The generated `hoi_cof_sft_data.json` contains:
- **Total samples**: 5,000
- **Referring (zoom)**: ~1,130 samples (multi-turn with tool calling)
- **Referring (direct)**: ~1,887 samples (single-turn, no tool)
- **Grounding (zoom)**: ~466 samples (multi-turn with tool calling)
- **Grounding (direct)**: ~1,517 samples (single-turn, no tool)

## Task Types

### Referring Task
Given person and object bounding boxes, predict the action.

**Input:**
```
Question: What action is the person performing with the object?
The person is located at [369, 262, 634, 951] and the object is at [498, 178, 629, 382].
```

**Output:** `holding baseball bat`

### Grounding Task
Given action and object category, locate all person-object pairs.

**Input:**
```
Question: Locate every person who is riding bicycle and the bicycle they interact with.
```

**Output:** `[{"bbox_2d": [x1, y1, x2, y2], "label": "person"}, {"bbox_2d": [...], "label": "bicycle"}]`

## Key Features

1. **1000x1000 Normalized Coordinates**: Compatible with Qwen3-VL
2. **Tool Name**: Uses `zoom_in` (verl-tool compatible)
3. **Zoomed Images**: Actual cropped images included for multi-turn samples
4. **Thinking Tags**: `<think>...</think>` for chain-of-thought reasoning
5. **Answer Tags**: `<answer>...</answer>` for structured output

## SFT Training

After generating the SFT dataset:

1. Create `dataset_info.json`:
```json
{
  "hoi_cof_sft": {
    "file_name": "hoi_cof_sft_data.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content", ...}
  }
}
```

2. Run SFT training:
```bash
bash examples/train/hoi/train_hoi_cof_sft.sh
```

---

# RL Data Preparation

## Overview

The `prepare_hoi.py` script converts HOI benchmark data into parquet format for RL training with verl-tool. The prompts are **aligned with SFT** to ensure the model can leverage SFT-learned behaviors during RL.

## Design Decisions

### Why Only `zoom_in` Tool?

Following the Chain-of-Focus paper and practical considerations:

1. **CoF Paper Design**: The original Chain-of-Focus uses only one zoom tool (`image_zoom_in_tool`)
2. **Progressive Learning**: Focus on mastering one tool first before adding complexity
3. **HOI Task Nature**: Most HOI tasks benefit from zooming into person-object regions
4. **SFT Alignment**: SFT data teaches `zoom_in` usage; RL should reinforce this

### Coordinate System

All bounding boxes use **1000x1000 normalized format**:
- Compatible with Qwen3-VL's native coordinate system
- Coordinates are integers from 0-1000
- Format: `[x1, y1, x2, y2]` where (x1, y1) is top-left, (x2, y2) is bottom-right

## RL Dataset Statistics

| Split | Grounding | Referring | Total |
|-------|-----------|-----------|-------|
| Train | 128,923 | 153,834 | 282,757 |
| Val | 1,000 | 1,000 | 2,000 |

**Data Sources:**
- HICO-DET (Human-Object Interaction Detection)
- SWIG-HOI (Situated Human-Object Interaction)

## Usage

### Prerequisites

```bash
pip install fire datasets
```

### Generate RL Training Data

```bash
python examples/data_preprocess/hoi/prepare_hoi.py \
    --local_dir data/hoi/train_data \
    --seed 42
```

### Output Files

```
data/hoi/train_data/
├── train.parquet    # 282,757 training samples
└── val.parquet      # 2,000 validation samples
```

### Parquet Schema

Each sample contains:
- `data_source`: Source dataset (hico_det, swig_hoi)
- `prompt`: List of messages `[{role, content}, ...]`
- `images`: List of image paths
- `ability`: Task type (grounding, referring)
- `reward_model`: Reward config for RL
- `extra_info`: Ground truth and metadata

## Prompt Format

### System Prompt

```
You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"zoom_in","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d). Coordinates use 1000x1000 normalized format.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2] in 1000x1000 normalized format..."},"target_image":{"type":"number","description":"The index of the image to zoom in on. Use 1 for the main image."}},"required":["bbox_2d", "target_image"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

### User Prompt (Referring Task)

```
<image> Question: What action is the person performing with the object?
The person is located at [501, 468, 607, 930] and the object is at [419, 436, 984, 957].
Respond with ONLY the action phrase in format: "{verb} {object}" (e.g., "riding bicycle", "holding cup"). Use base verb form, no articles.
Think in the mind first, and then decide whether to call tools one or more times OR provide final answer directly, using proper XML tags:
- Reasoning: <think>your thought process</think>
- Tool call: <tool_call>{"name": "zoom_in", "arguments": {...}}</tool_call>
- Final answer: <answer>your answer</answer>
```

### User Prompt (Grounding Task)

```
<image> Question: Locate every person who is riding bicycle and the bicycle they interact with.
For each person-object pair, output bbox coordinates in JSON format like: {"bbox_2d": [x1, y1, x2, y2], "label": "description"}. Coordinates should be in 1000x1000 normalized format.
Think in the mind first, and then decide whether to call tools one or more times OR provide final answer directly, using proper XML tags:
- Reasoning: <think>your thought process</think>
- Tool call: <tool_call>{"name": "zoom_in", "arguments": {...}}</tool_call>
- Final answer: <answer>your answer</answer>
```

## Files Modified for SFT/RL Alignment

The following files were updated to ensure SFT and RL use identical prompts and tools:

| File | Purpose | Changes |
|------|---------|---------|
| `prepare_hoi.py` | RL data generation | System prompt updated to match SFT; user instructions aligned |
| `verl_tool/servers/tools/hoi_detector.py` | Tool server | Removed `zoom_out`, `detect_objects`; only `zoom_in` available |
| `examples/train/hoi/train_hoi_qwen3vl_v2.sh` | RL training script | Updated echo messages to reflect single tool |

## RL Training

After generating the RL dataset:

```bash
# Start training
bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
```

The training script will:
1. Start the tool server (with `zoom_in` tool only)
2. Load the parquet data from `data/hoi/train_data/`
3. Run RL training with tool-use rewards

## Verification

To verify SFT/RL alignment:

```python
import pandas as pd
import json

# Load RL data
rl_data = pd.read_parquet('data/hoi/train_data/train.parquet')

# Load SFT data
with open('data/hoi_cof_sft/hoi_cof_sft_data.json', 'r') as f:
    sft_data = json.load(f)

# Compare system prompts
rl_system = rl_data.iloc[0]['prompt'][0]['content']
sft_system = sft_data[0]['messages'][0]['content']
assert rl_system == sft_system, "System prompts should match!"
print("✅ SFT and RL prompts are aligned!")
```

---

## References

- [Chain-of-Focus Paper](https://arxiv.org/abs/2505.15812)
- [verl-tool Repository](https://github.com/volcengine/verl-tool)
