# RL Tool Use Model Training Summary

## Model Architecture
- **Base Model**: Qwen3-VL-4B-Instruct (4B parameter vision-language model)
- **Training Method**: GRPO (Group Relative Policy Optimization) with agent-based tool use
- **Strategy**: FSDP2 distributed training with tensor parallelism (TP=2)
- **Hardware**: 8 GPUs, batch size 128

## Dataset
- **Source**: Combined HICO-DET and SWIG-HOI datasets
- **Tasks**: 
  - **Grounding**: Detect bounding boxes for person-object interaction pairs
  - **Referring**: Recognize actions given person and object bounding boxes
- **Format**: Parquet files (train.parquet, val.parquet) with image paths and annotations

### Complete Annotation Examples

#### Grounding Task Sample

```json
{
  "data_source": "hico_det",
  "ability": "grounding",
  "images": [
    {
      "image": "/workspace/verl-tool/data/hico_20160224_det/images/train2015/HICO_train2015_00016140.jpg"
    }
  ],
  "prompt": [
    {
      "role": "system",
      "content": "You are a helpful assistant for Human-Object Interaction detection.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\n  \"type\": \"function\",\n  \"function\": {\n    \"name\": \"zoom_in\",\n    \"description\": \"Zoom in on a specific region of the image to examine details.\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"bbox_2d\": {\n          \"type\": \"array\",\n          \"description\": \"Bounding box coordinates [x1, y1, x2, y2] in 1000x1000 normalized format.\",\n          \"items\": {\"type\": \"number\"}\n        },\n        \"target_image\": {\n          \"type\": \"number\",\n          \"description\": \"The index of the image to zoom in on. Use 1 for the main image.\"\n        }\n      },\n      \"required\": [\"bbox_2d\", \"target_image\"]\n    }\n  }\n}\n{\n  \"type\": \"function\",\n  \"function\": {\n    \"name\": \"zoom_out\",\n    \"description\": \"Reset the view to the original full image.\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"target_image\": {\n          \"type\": \"number\",\n          \"description\": \"The index of the image to reset. Use 1 for the main image.\"\n        }\n      },\n      \"required\": [\"target_image\"]\n    }\n  }\n}\n{\n  \"type\": \"function\",\n  \"function\": {\n    \"name\": \"detect_objects\",\n    \"description\": \"Detect objects in the image using Grounding DINO.\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"class_names\": {\n          \"type\": \"string\",\n          \"description\": \"Object classes to detect, separated by ' . ' (e.g., 'person . cup . chair').\"\n        },\n        \"target_image\": {\n          \"type\": \"number\",\n          \"description\": \"The index of the image to analyze. Use 1 for the main image.\"\n        },\n        \"confidence_threshold\": {\n          \"type\": \"number\",\n          \"description\": \"Minimum confidence for detections (0.0-1.0). Default: 0.25\"\n        }\n      },\n      \"required\": [\"class_names\", \"target_image\"]\n    }\n  }\n}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<image>Locate every person who is holding sandwich and the sandwich they interact with in this image. For each person-object pair, output bbox coordinates in JSON format like: {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description\"}\n\nGuidelines: Analyze the image to locate human-object interaction pairs. You may use zoom_in to examine details or detect_objects to find candidates. For each person-object pair, output bounding boxes in JSON format: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description\"}, ...]. Coordinates should be in the 1000x1000 normalized format."
    }
  ],
  "reward_model": {
    "ground_truth": "[{\"bbox_2d\": [23, 54, 270, 696], \"label\": \"person\"}, {\"bbox_2d\": [221, 57, 794, 894], \"label\": \"sandwich\"}]",
    "style": "rule",
    "task_type": "grounding"
  },
  "extra_info": {
    "action": "holding",
    "boxes_1000": [
      [23, 54, 270, 696],
      [221, 57, 794, 894]
    ],
    "dataset": "hico",
    "file_name": "HICO_train2015_00016140.jpg",
    "index": 29171,
    "num_pairs": 1,
    "object_box_idx": null,
    "object_category": "sandwich",
    "person_box_idx": null,
    "split": "train"
  }
}
```

**System Prompt (Formatted for Readability):**

```
You are a helpful assistant for Human-Object Interaction detection.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{
  "type": "function",
  "function": {
    "name": "zoom_in",
    "description": "Zoom in on a specific region of the image to examine details.",
    "parameters": {
      "type": "object",
      "properties": {
        "bbox_2d": {
          "type": "array",
          "description": "Bounding box coordinates [x1, y1, x2, y2] in 1000x1000 normalized format.",
          "items": {"type": "number"}
        },
        "target_image": {
          "type": "number",
          "description": "The index of the image to zoom in on. Use 1 for the main image."
        }
      },
      "required": ["bbox_2d", "target_image"]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "zoom_out",
    "description": "Reset the view to the original full image.",
    "parameters": {
      "type": "object",
      "properties": {
        "target_image": {
          "type": "number",
          "description": "The index of the image to reset. Use 1 for the main image."
        }
      },
      "required": ["target_image"]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "detect_objects",
    "description": "Detect objects in the image using Grounding DINO.",
    "parameters": {
      "type": "object",
      "properties": {
        "class_names": {
          "type": "string",
          "description": "Object classes to detect, separated by ' . ' (e.g., 'person . cup . chair')."
        },
        "target_image": {
          "type": "number",
          "description": "The index of the image to analyze. Use 1 for the main image."
        },
        "confidence_threshold": {
          "type": "number",
          "description": "Minimum confidence for detections (0.0-1.0). Default: 0.25"
        }
      },
      "required": ["class_names", "target_image"]
    }
  }
}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

**Key Fields:**
- **Task**: Locate person-object interaction pairs and output bounding boxes
- **Ground Truth**: JSON array with person and object bounding boxes in 1000x1000 normalized coordinates
- **Expected Output Format**: `[{"bbox_2d": [x1, y1, x2, y2], "label": "description"}, ...]`
- **Action**: "holding"
- **Object Category**: "sandwich"

#### Referring Task Sample

```json
{
  "data_source": "hico_det",
  "ability": "referring",
  "images": [
    {
      "image": "/workspace/verl-tool/data/hico_20160224_det/images/train2015/HICO_train2015_00038099.jpg"
    }
  ],
  "prompt": [
    {
      "role": "system",
      "content": "You are a helpful assistant for Human-Object Interaction detection.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\n  \"type\": \"function\",\n  \"function\": {\n    \"name\": \"zoom_in\",\n    \"description\": \"Zoom in on a specific region of the image to examine details.\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"bbox_2d\": {\n          \"type\": \"array\",\n          \"description\": \"Bounding box coordinates [x1, y1, x2, y2] in 1000x1000 normalized format.\",\n          \"items\": {\"type\": \"number\"}\n        },\n        \"target_image\": {\n          \"type\": \"number\",\n          \"description\": \"The index of the image to zoom in on. Use 1 for the main image.\"\n        }\n      },\n      \"required\": [\"bbox_2d\", \"target_image\"]\n    }\n  }\n}\n{\n  \"type\": \"function\",\n  \"function\": {\n    \"name\": \"zoom_out\",\n    \"description\": \"Reset the view to the original full image.\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"target_image\": {\n          \"type\": \"number\",\n          \"description\": \"The index of the image to reset. Use 1 for the main image.\"\n        }\n      },\n      \"required\": [\"target_image\"]\n    }\n  }\n}\n{\n  \"type\": \"function\",\n  \"function\": {\n    \"name\": \"detect_objects\",\n    \"description\": \"Detect objects in the image using Grounding DINO.\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"class_names\": {\n          \"type\": \"string\",\n          \"description\": \"Object classes to detect, separated by ' . ' (e.g., 'person . cup . chair').\"\n        },\n        \"target_image\": {\n          \"type\": \"number\",\n          \"description\": \"The index of the image to analyze. Use 1 for the main image.\"\n        },\n        \"confidence_threshold\": {\n          \"type\": \"number\",\n          \"description\": \"Minimum confidence for detections (0.0-1.0). Default: 0.25\"\n        }\n      },\n      \"required\": [\"class_names\", \"target_image\"]\n    }\n  }\n}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<image>Action Recognition Task: The first region {\"bbox_2d\": [501, 468, 607, 930], \"label\": \"person\"} contains a PERSON. The second region {\"bbox_2d\": [419, 436, 984, 957], \"label\": \"object\"} contains an OBJECT. Describe the action the person is performing with this object. Respond with only the action phrase (e.g., \"riding bicycle\", \"sitting on bench\").\n\nGuidelines: Analyze the provided bounding boxes to determine what action the person is performing with the object. You may use zoom_in to examine interaction details. Output only the action phrase (e.g., \"riding bicycle\", \"sitting on bench\"). Use base verb form without articles."
    }
  ],
  "reward_model": {
    "ground_truth": "loading truck",
    "style": "rule",
    "task_type": "referring"
  },
  "extra_info": {
    "action": null,
    "boxes_1000": [
      [44, 346, 139, 509],
      [8, 483, 377, 836],
      [6, 524, 66, 832],
      [576, 225, 718, 460],
      [419, 436, 984, 957],
      [501, 468, 607, 930],
      [211, 476, 285, 679],
      [8, 462, 391, 865],
      [906, 407, 989, 593],
      [423, 444, 992, 947],
      [257, 431, 336, 663],
      [853, 389, 948, 677],
      [721, 215, 883, 581]
    ],
    "dataset": "hico",
    "file_name": "HICO_train2015_00038099.jpg",
    "index": 92951,
    "num_pairs": null,
    "object_box_idx": 4,
    "object_category": null,
    "person_box_idx": 5,
    "split": "train"
  }
}
```

**System Prompt (Formatted for Readability):**

```
You are a helpful assistant for Human-Object Interaction detection.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{
  "type": "function",
  "function": {
    "name": "zoom_in",
    "description": "Zoom in on a specific region of the image to examine details.",
    "parameters": {
      "type": "object",
      "properties": {
        "bbox_2d": {
          "type": "array",
          "description": "Bounding box coordinates [x1, y1, x2, y2] in 1000x1000 normalized format.",
          "items": {"type": "number"}
        },
        "target_image": {
          "type": "number",
          "description": "The index of the image to zoom in on. Use 1 for the main image."
        }
      },
      "required": ["bbox_2d", "target_image"]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "zoom_out",
    "description": "Reset the view to the original full image.",
    "parameters": {
      "type": "object",
      "properties": {
        "target_image": {
          "type": "number",
          "description": "The index of the image to reset. Use 1 for the main image."
        }
      },
      "required": ["target_image"]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "detect_objects",
    "description": "Detect objects in the image using Grounding DINO.",
    "parameters": {
      "type": "object",
      "properties": {
        "class_names": {
          "type": "string",
          "description": "Object classes to detect, separated by ' . ' (e.g., 'person . cup . chair')."
        },
        "target_image": {
          "type": "number",
          "description": "The index of the image to analyze. Use 1 for the main image."
        },
        "confidence_threshold": {
          "type": "number",
          "description": "Minimum confidence for detections (0.0-1.0). Default: 0.25"
        }
      },
      "required": ["class_names", "target_image"]
    }
  }
}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

**Key Fields:**
- **Task**: Given person and object bounding boxes, recognize the action
- **Ground Truth**: Action phrase string (e.g., "loading truck")
- **Expected Output Format**: Action phrase only (verb + object, no articles)
- **Person Bbox Index**: 5 (in boxes_1000 array: [501, 468, 607, 930])
- **Object Bbox Index**: 4 (in boxes_1000 array: [419, 436, 984, 957])

## Tool Use Capabilities
The model was trained with agent-based tool calling:
- **zoom_in**: Zoom into specific image regions for detailed examination
- **zoom_out**: Reset to full image view
- **detect_objects**: Object detection using Grounding DINO
- **Max turns**: 3 tool interaction turns per episode

## Training Configuration
- **Algorithm**: GRPO with n=8 samples per prompt
- **Learning rate**: 1e-6 (actor), 1e-5 (critic)
- **Sequence lengths**: Max prompt 16,384 tokens, max response 8,192 tokens
- **Training steps**: 100 steps total
- **Checkpoint frequency**: Every 10 steps
- **Final checkpoint**: Step 100

## Reward Function
- **Grounding tasks**: Binary reward (1.0 if IoU ≥ 0.5 for all ground truth boxes, else 0.0)
- **Referring tasks**: Word overlap score (1.0 for exact match, else word overlap ratio)

## Results
- **Checkpoint location**: `checkpoints/hoi_reward/hoi_reward-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b64-t1.0-lr1e-6-hoi-detection/`
- **Final checkpoint**: `global_step_100/actor/huggingface/` (ready for deployment)
- **Training completed**: Successfully trained for 100 steps with multi-turn RL enabled

