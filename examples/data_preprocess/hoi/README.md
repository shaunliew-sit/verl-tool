# HOI Chain-of-Focus SFT Data Generation

This directory contains scripts for generating HOI (Human-Object Interaction) datasets for training.

## Overview

The Chain-of-Focus (CoF) approach teaches models to:
1. **Think** before acting (`<think>...</think>` tags)
2. **Use tools** when needed (`<tool_call>...</tool_call>` tags)
3. **Provide structured answers** (`<answer>...</answer>` tags)

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

Converts benchmark data to parquet format for RL training.

### `prepare_sft_data.py` - Alternative SFT Data

Alternative SFT preparation (less recommended, use `generate_hoi_cof_sft.py` instead).

## Dataset Statistics

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

## Training

After generating the dataset:

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

## References

- [Chain-of-Focus Paper](https://arxiv.org/abs/2505.15812)
- [verl-tool Repository](https://github.com/volcengine/verl-tool)
