#!/usr/bin/env python3
"""
Prepare SFT (Supervised Fine-Tuning) dataset for HOI detection.

This script creates a dataset with ideal input-output pairs to teach the model:
1. The expected output format (short action phrases for referring, bboxes for grounding)
2. How to use tools appropriately
3. How to structure the reasoning

SFT before RL is recommended because:
- The base model doesn't know the task format
- RL needs reasonable initial behavior to improve upon
- SFT provides the "format" while RL optimizes for "correctness"

Usage:
    python examples/data_preprocess/hoi/prepare_sft_data.py

Output:
    data/hoi/sft_data/train.jsonl - Training data for SFT
    data/hoi/sft_data/eval.jsonl  - Evaluation data for SFT
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Import the same system prompt and guidelines from the RL preparation
try:
    from prepare_hoi import SYSTEM_PROMPT, GROUNDING_GUIDELINE, REFERRING_GUIDELINE
except ImportError:
    # Define here if import fails
    SYSTEM_PROMPT = '''You are an expert at analyzing human-object interactions in images. You have access to the following tools to help with your analysis:

<tools>
[
    {
        "type": "function",
        "function": {
            "name": "zoom_in",
            "description": "Zoom in on a specific region of the current image to see more details. Use this when you need to examine a specific area more closely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Bounding box [x1, y1, x2, y2] in pixel coordinates specifying the region to zoom into."
                    }
                },
                "required": ["bbox"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_out",
            "description": "Reset to the original full image view. Use this after zooming in to see the full context again.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": "Detect objects in the current view using Grounding DINO. Returns bounding boxes for detected objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_names": {
                        "type": "string",
                        "description": "Object classes to detect, separated by ' . ' (e.g., 'person . cup . chair')"
                    }
                },
                "required": ["class_names"]
            }
        }
    }
]
</tools>

When you need to use a tool, format your request as:
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>

After analyzing the image, provide your final answer.'''

    GROUNDING_GUIDELINE = """## HOI Grounding Task

For the interaction "{action}", locate the person and object involved.

Instructions:
1. Find the person performing this action
2. Find the object they are interacting with
3. Output both bounding boxes

Output format:
[{{"bbox_2d": [x1, y1, x2, y2], "label": "person"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "{object}"}}]"""

    REFERRING_GUIDELINE = """## HOI Referring Task

The first region {person_bbox} contains a PERSON.
The second region {object_bbox} contains {object_name}.

Task: Describe what action the person is performing with this object.

Output only the action phrase (e.g., "riding bicycle", "sitting on bench").
Use base verb form without articles."""


def create_referring_sft_example(
    image_path: str,
    person_bbox: List[int],
    object_bbox: List[int],
    object_category: str,
    action: str,
    use_tool: bool = False
) -> Dict[str, Any]:
    """
    Create a single SFT example for the referring task.
    
    Args:
        image_path: Path to the image
        person_bbox: Person bounding box [x1, y1, x2, y2]
        object_bbox: Object bounding box [x1, y1, x2, y2]
        object_category: Category of the object
        action: Ground truth action phrase
        use_tool: Whether to include tool usage in the response
        
    Returns:
        SFT example with messages format
    """
    # Create user message
    user_content = REFERRING_GUIDELINE.format(
        person_bbox=json.dumps({"bbox_2d": person_bbox, "label": "person"}),
        object_bbox=json.dumps({"bbox_2d": object_bbox, "label": object_category}),
        object_name=object_category
    )
    
    # Create ideal assistant response
    if use_tool:
        # Example with tool usage - model zooms in to examine interaction
        interaction_center = [
            (person_bbox[0] + object_bbox[0]) // 2,
            (person_bbox[1] + object_bbox[1]) // 2,
            (person_bbox[2] + object_bbox[2]) // 2,
            (person_bbox[3] + object_bbox[3]) // 2
        ]
        assistant_response = f"""I'll examine the interaction between the person and {object_category}.

<tool_call>
{{"name": "zoom_in", "arguments": {{"bbox": {interaction_center}}}}}
</tool_call>"""
        
        # After tool use, provide the answer
        tool_observation = "Zoomed in view shows the interaction area in detail."
        final_response = f"Based on my analysis, the person is {action}."
        
        # Create multi-turn conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_content}
            ]},
            {"role": "assistant", "content": assistant_response},
            {"role": "user", "content": tool_observation},
            {"role": "assistant", "content": final_response}
        ]
    else:
        # Direct response without tool
        assistant_response = action
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_content}
            ]},
            {"role": "assistant", "content": assistant_response}
        ]
    
    return {
        "messages": messages,
        "task_type": "referring",
        "ground_truth": action
    }


def create_grounding_sft_example(
    image_path: str,
    action: str,
    object_category: str,
    person_bbox: List[int],
    object_bbox: List[int],
    use_tool: bool = False
) -> Dict[str, Any]:
    """
    Create a single SFT example for the grounding task.
    
    Args:
        image_path: Path to the image
        action: Action phrase describing the interaction
        object_category: Category of the object
        person_bbox: Ground truth person bounding box
        object_bbox: Ground truth object bounding box
        use_tool: Whether to include tool usage
        
    Returns:
        SFT example with messages format
    """
    # Create user message
    user_content = GROUNDING_GUIDELINE.format(
        action=action,
        object=object_category
    )
    
    # Create ideal assistant response (the correct bounding boxes)
    ground_truth = [
        {"bbox_2d": person_bbox, "label": "person"},
        {"bbox_2d": object_bbox, "label": object_category}
    ]
    
    if use_tool:
        # Use detect_objects tool first
        assistant_response = f"""I'll use object detection to find the person and {object_category}.

<tool_call>
{{"name": "detect_objects", "arguments": {{"class_names": "person . {object_category}"}}}}
</tool_call>"""
        
        # Simulated tool observation
        tool_observation = f"""Detected 2 objects:
- person: bbox={person_bbox}, confidence=0.92
- {object_category}: bbox={object_bbox}, confidence=0.88"""
        
        final_response = f"Based on the detection, the person performing '{action}' is at {person_bbox} and the {object_category} is at {object_bbox}.\n\n{json.dumps(ground_truth)}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_content}
            ]},
            {"role": "assistant", "content": assistant_response},
            {"role": "user", "content": tool_observation},
            {"role": "assistant", "content": final_response}
        ]
    else:
        # Direct response
        assistant_response = json.dumps(ground_truth)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_content}
            ]},
            {"role": "assistant", "content": assistant_response}
        ]
    
    return {
        "messages": messages,
        "task_type": "grounding",
        "ground_truth": json.dumps(ground_truth)
    }


def load_hico_annotations(hico_dir: str) -> List[Dict]:
    """Load HICO-DET annotations."""
    import scipy.io as sio
    
    anno_file = os.path.join(hico_dir, "anno.mat")
    if not os.path.exists(anno_file):
        print(f"Warning: anno.mat not found at {anno_file}")
        return []
    
    anno = sio.loadmat(anno_file)
    
    # Load action/object mappings
    hoi_file = os.path.join(hico_dir, "hoi_list.json")
    if os.path.exists(hoi_file):
        with open(hoi_file) as f:
            hoi_list = json.load(f)
    else:
        # Fallback - would need proper HICO-DET metadata
        print("Warning: hoi_list.json not found, using default mappings")
        return []
    
    return hoi_list


def main():
    """Main function to prepare SFT data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SFT data for HOI detection")
    parser.add_argument("--hico-dir", type=str, default="data/hico_20160224_det",
                        help="Path to HICO-DET dataset")
    parser.add_argument("--output-dir", type=str, default="data/hoi/sft_data",
                        help="Output directory for SFT data")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum number of SFT samples to generate")
    parser.add_argument("--tool-usage-ratio", type=float, default=0.3,
                        help="Ratio of examples that include tool usage")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if RL training data exists (we can convert it to SFT format)
    rl_train_file = "data/hoi/train_data/train.parquet"
    rl_val_file = "data/hoi/train_data/val.parquet"
    
    if os.path.exists(rl_train_file):
        print(f"Found existing RL training data at {rl_train_file}")
        print("Converting to SFT format...")
        
        import pandas as pd
        
        # Load RL data
        train_df = pd.read_parquet(rl_train_file)
        val_df = pd.read_parquet(rl_val_file)
        
        def convert_to_sft(row) -> Optional[Dict]:
            """Convert a single RL row to SFT format."""
            try:
                # Parse the data_source to get task info
                reward_model = json.loads(row['reward_model']) if isinstance(row['reward_model'], str) else row['reward_model']
                task_type = reward_model.get('task_type', 'referring')
                ground_truth = reward_model.get('ground_truth', '')
                
                # Get the prompt (contains the task instruction)
                prompt = row['prompt'] if 'prompt' in row else ''
                
                # Get extra info for bounding boxes
                extra_info = json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row.get('extra_info', {})
                
                # Get image path
                images = json.loads(row['images']) if isinstance(row['images'], str) else row.get('images', [])
                image_path = images[0] if images else ''
                
                # Decide if this example should use tools
                use_tool = random.random() < args.tool_usage_ratio
                
                if task_type == 'referring':
                    # Extract bounding box info from extra_info or prompt
                    person_bbox = extra_info.get('person_bbox', [100, 100, 300, 400])
                    object_bbox = extra_info.get('object_bbox', [200, 150, 400, 500])
                    object_category = extra_info.get('object_category', 'object')
                    
                    return create_referring_sft_example(
                        image_path=image_path,
                        person_bbox=person_bbox,
                        object_bbox=object_bbox,
                        object_category=object_category,
                        action=ground_truth,
                        use_tool=use_tool
                    )
                else:  # grounding
                    # Parse ground truth for bounding boxes
                    gt_boxes = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
                    if len(gt_boxes) >= 2:
                        person_bbox = gt_boxes[0].get('bbox_2d', [100, 100, 300, 400])
                        object_bbox = gt_boxes[1].get('bbox_2d', [200, 150, 400, 500])
                        object_category = gt_boxes[1].get('label', 'object')
                    else:
                        return None
                    
                    action = extra_info.get('action', 'interacting with')
                    
                    return create_grounding_sft_example(
                        image_path=image_path,
                        action=action,
                        object_category=object_category,
                        person_bbox=person_bbox,
                        object_bbox=object_bbox,
                        use_tool=use_tool
                    )
            except Exception as e:
                print(f"Warning: Failed to convert row: {e}")
                return None
        
        # Convert training data
        train_sft = []
        for _, row in train_df.iterrows():
            if len(train_sft) >= args.max_samples:
                break
            sft_example = convert_to_sft(row)
            if sft_example:
                train_sft.append(sft_example)
        
        # Convert validation data
        val_sft = []
        for _, row in val_df.iterrows():
            if len(val_sft) >= args.max_samples // 10:  # 10% of training size
                break
            sft_example = convert_to_sft(row)
            if sft_example:
                val_sft.append(sft_example)
        
        # Save SFT data
        train_output = os.path.join(args.output_dir, "train.jsonl")
        with open(train_output, 'w') as f:
            for example in train_sft:
                f.write(json.dumps(example) + '\n')
        print(f"Saved {len(train_sft)} training examples to {train_output}")
        
        val_output = os.path.join(args.output_dir, "eval.jsonl")
        with open(val_output, 'w') as f:
            for example in val_sft:
                f.write(json.dumps(example) + '\n')
        print(f"Saved {len(val_sft)} validation examples to {val_output}")
        
    else:
        print(f"RL training data not found at {rl_train_file}")
        print("Please run prepare_hoi.py first to generate training data")
        print("Then run this script to convert to SFT format")
        return
    
    print("\n" + "="*60)
    print("SFT Data Preparation Complete!")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Training examples: {len(train_sft)}")
    print(f"Validation examples: {len(val_sft)}")
    print(f"Tool usage ratio: {args.tool_usage_ratio:.0%}")
    print("\nTo run SFT training, use:")
    print("  bash examples/train/hoi/train_hoi_sft.sh")
    print("\nAfter SFT, run RL training with:")
    print("  bash examples/train/hoi/train_hoi_qwen3vl_v2.sh")


if __name__ == "__main__":
    main()

