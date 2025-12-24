# HOI Detection RL Training

Human-Object Interaction (HOI) detection using Reinforcement Learning with VTool-style outcome-based rewards.

## Overview

This implementation adapts the pixel_reasoner RL training framework for HOI detection tasks:

- **Grounding Task**: Given an action and object description, locate the person-object interaction pairs
- **Referring Task**: Given bounding boxes of a person and object, predict the action being performed

The training uses the VTool paradigm with pure outcome-based rewards - no intermediate penalties for tool usage patterns.

## Changes from pixel_reasoner

### Tools (`verl_tool/servers/tools/hoi_detector.py`)

| Original (pixel_reasoner) | HOI Implementation | Notes |
|---------------------------|-------------------|-------|
| `crop_image` / `zoom_in` | `zoom_in` | Kept - crop image region |
| `select_frames` | **Removed** | Not needed for single images |
| - | `zoom_out` | **Added** - reset to original view |
| - | `detect_objects` | **Added** - Grounding DINO detection |

### Reward Manager (`verl_tool/workers/reward_manager/hoi_reward.py`)

| Original (pixel_reasoner) | HOI Implementation | Notes |
|---------------------------|-------------------|-------|
| `math_verify` scoring | IoU + BERTScore | Task-specific scoring |
| Curiosity penalty | **Removed** | VTool paradigm |
| Action redundancy penalty | **Removed** | VTool paradigm |

**Reward Functions:**

- **Grounding**: Binary IoU threshold
  - Score = 1.0 if predicted boxes match GT with IoU >= 0.5
  - Score = 0.0 otherwise

- **Referring**: Hybrid BERTScore
  - Score = 1.0 for exact match
  - Score = BERTScore F1 otherwise (using `microsoft/deberta-v2-xxlarge-mnli`)

### Dataset Format

| Column | Type | Description |
|--------|------|-------------|
| `data_source` | string | "hico_det" or "swig_det" |
| `prompt` | list | System prompt with tools + user query |
| `images` | list | Image path dictionaries |
| `ability` | string | "grounding" or "referring" |
| `reward_model` | dict | Ground truth and task type |
| `extra_info` | dict | Metadata (boxes, action, etc.) |

## Prerequisites

### Dependencies

```bash
# Core dependencies (should already be installed)
pip install transformers torch

# BERTScore for referring task reward
pip install bert_score

# Grounding DINO will auto-download on first use
# Model: IDEA-Research/grounding-dino-tiny
```

### Data Preparation

1. Ensure benchmark data is available:
   - `data/benchmarks_simplified/` - Simplified JSON annotations
   - `data/hico_20160224_det/images/` - HICO-DET images
   - `data/swig_hoi/images_512/` - SWIG-HOI images

2. Run data preparation:
   ```bash
   python examples/data_preprocess/hoi/prepare_hoi.py \
       --local_dir data/hoi/train_data \
       --seed 42
   ```

3. Verify the generated parquet files:
   ```bash
   python view_parquet.py data/hoi/train_data/train.parquet --rows 5 --show-all
   ```

## Training

### Quick Start

```bash
# Make the script executable
chmod +x examples/train/hoi/train_hoi_qwen3vl.sh

# Run training
bash examples/train/hoi/train_hoi_qwen3vl.sh
```

### Configuration

Key parameters in `train_hoi_qwen3vl.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | Qwen/Qwen3-VL-4B-Instruct | Base model |
| `reward_manager` | hoi_reward | Reward function |
| `tool_type` | hoi_detector | Tool server type |
| `max_turns` | 3 | Max tool interaction turns |
| `batch_size` | 128 | Training batch size |
| `lr` | 1e-6 | Learning rate |
| `temperature` | 1.0 | Sampling temperature |

### Multi-GPU Training

The script is configured for 8 GPUs by default. Adjust these parameters for your setup:

```bash
n_gpus_per_node=8
n_nodes=1
tensor_model_parallel_size=2
```

## File Structure

```
examples/
├── data_preprocess/
│   └── hoi/
│       └── prepare_hoi.py           # Dataset preparation
├── train/
│   └── hoi/
│       ├── README.md                # This file
│       └── train_hoi_qwen3vl.sh     # Training script
verl_tool/
├── servers/tools/
│   └── hoi_detector.py              # HOI tools
├── workers/reward_manager/
│   └── hoi_reward.py                # HOI rewards
data/
└── hoi/
    └── train_data/
        ├── train.parquet            # Training data (282,757 samples)
        └── val.parquet              # Validation data (800 samples)
```

## Reward Functions Explained

### Grounding Task (Binary IoU)

```python
def grounding_score(pred_boxes, gt_boxes, iou_threshold=0.5):
    # For each GT box, find best matching prediction
    # Return 1.0 if all GT boxes matched with IoU >= 0.5
    # Return 0.0 otherwise
```

This aligns with the AR@0.5 evaluation metric used for HOI detection.

### Referring Task (Hybrid BERTScore)

```python
def referring_score(prediction, ground_truth):
    pred_clean = clean_text(prediction).lower()
    gt_clean = clean_text(ground_truth).lower()
    
    # Exact match gets full reward
    if pred_clean == gt_clean:
        return 1.0
    
    # Otherwise use BERTScore F1 for semantic similarity
    return bertscore_f1(pred_clean, gt_clean)
```

This handles vocabulary variations (e.g., "riding bicycle" vs "riding a bicycle") by computing semantic similarity.

## Evaluation

After training, evaluate using the standard HOI evaluation scripts:

### Grounding Evaluation
- Metric: AR (Average Recall) at IoU thresholds 0.5-0.95
- Key metrics: AR@0.5, AR@0.75

### Referring Evaluation
- Metrics: METEOR, CIDEr, BERTScore F1
- Also reports exact match accuracy

## Next Steps

1. **Hyperparameter Tuning**
   - Experiment with different learning rates (1e-6 to 1e-5)
   - Try different temperature values for exploration
   - Adjust max_turns for tool usage patterns

2. **Model Scaling**
   - Try larger models: Qwen3-VL-8B-Instruct
   - Compare with other VLMs

3. **Reward Engineering**
   - Consider weighted combination of IoU scores
   - Experiment with continuous IoU reward vs binary

4. **Tool Usage Analysis**
   - Monitor tool usage patterns during training
   - Analyze when zoom_in/detect_objects are most beneficial

5. **Evaluation Extensions**
   - Add mAP metric for grounding
   - Compare with supervised baselines

## Troubleshooting

### BERTScore Model Loading

The BERTScore model (`microsoft/deberta-v2-xxlarge-mnli`) requires ~12-16GB GPU memory. If you encounter OOM issues:

1. Set a specific GPU for reward computation:
   ```python
   export CUDA_VISIBLE_DEVICES=0
   ```

2. Or use a smaller model by modifying `hoi_reward.py`:
   ```python
   model_type="roberta-large"  # ~4GB instead
   ```

### Grounding DINO Loading

The Grounding DINO model downloads on first use. Ensure internet access or pre-download:

```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
```

### Image Path Issues

If images are not found, verify paths in the parquet file:
```bash
python view_parquet.py data/hoi/train_data/train.parquet --rows 1 --show-all | grep image
```

Ensure the image directories match your setup:
- HICO: `data/hico_20160224_det/images/train2015/` and `test2015/`
- SWIG: `data/swig_hoi/images_512/`

## References

- [VTool-R1](https://github.com/VTool-R1/VTool-R1) - VTool paradigm for outcome-based rewards
- [Pixel-Reasoner](https://github.com/TIGER-AI-Lab/Pixel-Reasoner) - Base implementation
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) - Training framework

