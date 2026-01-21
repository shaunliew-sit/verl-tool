# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare HOI (Human-Object Interaction) dataset for RL training with MULTI-TOOL support.

This script converts the simplified benchmark JSON files to parquet format
compatible with the verl-tool training pipeline. It includes:
- Multi-tool system prompt (zoom_in, zoom_out, detect_objects)
- refer_boxes column for spatial linking (3 boxes: person, object, interaction)
- Relative image paths for portability

Usage:
    python examples/data_preprocess/hoi/prepare_hoi_multitool.py \
        --local_dir data/hoi/train_data_multitool \
        --seed 42

Input files (from data/benchmarks_simplified/):
    Training:
        - hico_ground_train_simplified.json
        - hico_referring_train_simplified.json
        - swig_ground_train_simplified.json
        - swig_referring_train_simplified.json
    Validation (from test files):
        - hico_ground_test_simplified.json
        - hico_action_referring_test_simplified.json
        - swig_ground_test_simplified.json
        - swig_action_referring_test_simplified.json

Output:
    - train.parquet: Combined training data with refer_boxes + multi-tool
    - val.parquet: Validation data with refer_boxes + multi-tool
"""

import fire
import os
import json
import datasets
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional

# =============================================================================
# System Prompt - MULTI-TOOL version with zoom_in, zoom_out, detect_objects
# =============================================================================
SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"zoom_in","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d). Coordinates use 1000x1000 normalized format.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2] in 1000x1000 normalized format."},"target_image":{"type":"number","description":"The index of the image to zoom in on. Use 1 for the main image."}},"required":["bbox_2d", "target_image"], "type":"object"}}}
{"type": "function", "function": {"name":"zoom_out","description":"Reset the view to the original full image.","parameters":{"properties":{"target_image":{"type":"number","description":"The index of the image. Use 1 for the main image."}},"required":["target_image"], "type":"object"}}}
{"type": "function", "function": {"name":"detect_objects","description":"Detect objects in the image using Grounding DINO.","parameters":{"properties":{"class_names":{"type":"string","description":"Classes to detect, separated by ' . ' (e.g., 'person . bicycle')"},"target_image":{"type":"number","description":"The index of the image. Use 1 for the main image."}},"required":["class_names", "target_image"], "type":"object"}}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# User instruction suffix - aligned with SFT data
USER_INSTRUCTION_SUFFIX = "Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."

# Guidelines aligned with SFT format
GROUNDING_GUIDELINE = f"""For each person-object pair, output bbox coordinates in JSON format like: {{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}. Coordinates should be in 1000x1000 normalized format.
{USER_INSTRUCTION_SUFFIX}"""

REFERRING_GUIDELINE = f"""Respond with ONLY the action phrase in format: "{{verb}} {{object}}" (e.g., "riding bicycle", "holding cup"). Use base verb form, no articles.
{USER_INSTRUCTION_SUFFIX}"""

# Image directory mapping (relative paths)
IMAGE_DIRS = {
    'hico_train': 'data/hico_20160224_det/images/train2015',
    'hico_test': 'data/hico_20160224_det/images/test2015',
    'swig': 'data/swig_hoi/images_512',
}

# Benchmark files
BENCHMARK_FILES = {
    'train': {
        'hico_ground': 'data/benchmarks_simplified/hico_ground_train_simplified.json',
        'hico_referring': 'data/benchmarks_simplified/hico_referring_train_simplified.json',
        'swig_ground': 'data/benchmarks_simplified/swig_ground_train_simplified.json',
        'swig_referring': 'data/benchmarks_simplified/swig_referring_train_simplified.json',
    },
    'val': {
        'hico_ground': 'data/benchmarks_simplified/hico_ground_test_simplified.json',
        'hico_referring': 'data/benchmarks_simplified/hico_action_referring_test_simplified.json',
        'swig_ground': 'data/benchmarks_simplified/swig_ground_test_simplified.json',
        'swig_referring': 'data/benchmarks_simplified/swig_action_referring_test_simplified.json',
    }
}


def get_image_path(file_name: str, dataset_type: str, split: str) -> str:
    """
    Get the RELATIVE image path based on dataset type and split.
    
    Args:
        file_name: Image filename
        dataset_type: 'hico' or 'swig'
        split: 'train' or 'val'
        
    Returns:
        Relative path to the image (for portability)
    """
    if dataset_type == 'hico':
        if split == 'train':
            base_dir = IMAGE_DIRS['hico_train']
        else:
            base_dir = IMAGE_DIRS['hico_test']
    else:  # swig
        base_dir = IMAGE_DIRS['swig']
    
    # Return relative path (not absolute) for portability
    return f"{base_dir}/{file_name}"


def convert_boxes_to_1000(boxes: List[List[float]], width: int, height: int) -> List[List[int]]:
    """Convert pixel coordinates to 1000x1000 normalized format."""
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box
        result.append([
            int(x1 / (width - 1) * 1000) if width > 1 else int(x1),
            int(y1 / (height - 1) * 1000) if height > 1 else int(y1),
            int(x2 / (width - 1) * 1000) if width > 1 else int(x2),
            int(y2 / (height - 1) * 1000) if height > 1 else int(y2),
        ])
    return result


def compute_interaction_box(person_box: List[int], object_box: List[int]) -> List[int]:
    """
    Compute union box of person and object (interaction region) in 1000-grid format.
    
    Args:
        person_box: [x1, y1, x2, y2] in 0-1000 format
        object_box: [x1, y1, x2, y2] in 0-1000 format
        
    Returns:
        Union box [x1, y1, x2, y2] in 0-1000 format
    """
    return [
        min(person_box[0], object_box[0]),
        min(person_box[1], object_box[1]),
        max(person_box[2], object_box[2]),
        max(person_box[3], object_box[3])
    ]


def process_grounding_sample(sample: Dict[str, Any], dataset_type: str, split: str, idx: int) -> Dict[str, Any]:
    """
    Process a grounding sample into RL format with refer_boxes for spatial linking.
    
    Grounding task: Given query about action+object, output bounding boxes.
    Handles both simplified format (with query/response) and raw format (without).
    """
    file_name = sample['file_name']
    image_path = get_image_path(file_name, dataset_type, split)
    
    # Get or compute boxes_1000
    if 'boxes_1000' in sample:
        boxes_1000 = sample['boxes_1000']
    else:
        boxes_1000 = convert_boxes_to_1000(
            sample.get('boxes', []),
            sample.get('width', 1000),
            sample.get('height', 1000)
        )
    
    # Get action and object category
    action = sample.get('action', 'interacting with')
    object_category = sample.get('object_category', 'object')
    
    # Handle response
    if 'response' in sample:
        response = sample['response']
    else:
        # Build response from boxes (alternating person/object)
        gt_box_inds = sample.get('gt_box_inds', list(range(len(boxes_1000))))
        response_items = []
        for i in range(0, len(gt_box_inds), 2):
            if i < len(gt_box_inds):
                person_idx = gt_box_inds[i]
                if person_idx < len(boxes_1000):
                    response_items.append({"bbox_2d": boxes_1000[person_idx], "label": "person"})
            if i + 1 < len(gt_box_inds):
                object_idx = gt_box_inds[i + 1]
                if object_idx < len(boxes_1000):
                    response_items.append({"bbox_2d": boxes_1000[object_idx], "label": object_category})
        response = json.dumps(response_items)
    
    # Build user content - aligned with SFT format
    query = f"Question: Locate every person who is {action} {object_category} and the {object_category} they interact with."
    user_content = f"<image> {query}\n{GROUNDING_GUIDELINE}"
    
    # Keep ground truth as string (JSON) for consistency
    ground_truth = response if isinstance(response, str) else json.dumps(response)
    
    # Compute refer_boxes for spatial linking (first person-object pair)
    gt_box_inds = sample.get('gt_box_inds', list(range(len(boxes_1000))))
    if len(gt_box_inds) >= 2 and len(boxes_1000) >= 2:
        p_idx, o_idx = gt_box_inds[0], gt_box_inds[1]
        person_box = boxes_1000[p_idx] if p_idx < len(boxes_1000) else [0, 0, 500, 500]
        obj_box = boxes_1000[o_idx] if o_idx < len(boxes_1000) else [500, 500, 1000, 1000]
        interaction_box = compute_interaction_box(person_box, obj_box)
        refer_boxes = [person_box, obj_box, interaction_box]
    else:
        # Fallback if not enough boxes
        refer_boxes = [[0, 0, 500, 500], [500, 500, 1000, 1000], [0, 0, 1000, 1000]]
    
    return {
        "data_source": f"{dataset_type}_det",
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            }
        ],
        "images": [{"image": image_path}],
        "ability": "grounding",
        "refer_boxes": refer_boxes,  # Added for spatial linking
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
            "task_type": "grounding",
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "file_name": file_name,
            "dataset": dataset_type,
            "boxes_1000": boxes_1000,
            "action": sample.get('action', ''),
            "object_category": sample.get('object_category', ''),
            "num_pairs": sample.get('num_pairs', 1),
        }
    }


def process_referring_sample(sample: Dict[str, Any], dataset_type: str, split: str, idx: int) -> Dict[str, Any]:
    """
    Process a referring sample into RL format with refer_boxes for spatial linking.
    
    Referring task: Given person and object bounding boxes, predict the action.
    Handles both simplified format (with query/response) and raw format (without).
    """
    file_name = sample['file_name']
    image_path = get_image_path(file_name, dataset_type, split)
    
    # Get or compute boxes_1000
    if 'boxes_1000' in sample:
        boxes_1000 = sample['boxes_1000']
    else:
        boxes_1000 = convert_boxes_to_1000(
            sample.get('boxes', []),
            sample.get('width', 1000),
            sample.get('height', 1000)
        )
    
    # Get box indices and response
    person_box_idx = sample.get('person_box_idx', 0)
    object_box_idx = sample.get('object_box_idx', 1)
    
    # Get boxes
    person_box = boxes_1000[person_box_idx] if person_box_idx < len(boxes_1000) else [0, 0, 1000, 1000]
    object_box = boxes_1000[object_box_idx] if object_box_idx < len(boxes_1000) else [0, 0, 1000, 1000]
    
    # Get response (action phrase)
    if 'response' in sample:
        response = sample['response']
    else:
        response = sample.get('gt_action', '')
    
    # Build user content - aligned with SFT format
    query = f"Question: What action is the person performing with the object?\nThe person is located at {person_box} and the object is at {object_box}."
    user_content = f"<image> {query}\n{REFERRING_GUIDELINE}"
    
    # Compute interaction box for spatial linking
    interaction_box = compute_interaction_box(person_box, object_box)
    refer_boxes = [person_box, object_box, interaction_box]
    
    return {
        "data_source": f"{dataset_type}_det",
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            }
        ],
        "images": [{"image": image_path}],
        "ability": "referring",
        "refer_boxes": refer_boxes,  # Added for spatial linking
        "reward_model": {
            "style": "rule",
            "ground_truth": response,
            "task_type": "referring",
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "file_name": file_name,
            "dataset": dataset_type,
            "boxes_1000": boxes_1000,
            "person_box_idx": sample.get('person_box_idx', 0),
            "object_box_idx": sample.get('object_box_idx', 1),
        }
    }


def load_and_process_file(
    file_path: str,
    dataset_type: str,
    task_type: str,
    split: str,
    max_samples: Optional[int] = None,
    check_images: bool = True
) -> List[Dict[str, Any]]:
    """
    Load a JSON file and process all samples.
    
    Args:
        file_path: Path to JSON file
        dataset_type: 'hico' or 'swig'
        task_type: 'ground' or 'referring'
        split: 'train' or 'val'
        max_samples: Optional limit on samples
        check_images: Whether to verify image files exist
        
    Returns:
        List of processed samples
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []
    
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    processed = []
    skipped = 0
    for idx, sample in enumerate(data):
        if task_type == 'ground':
            processed_sample = process_grounding_sample(sample, dataset_type, split, idx)
        else:
            processed_sample = process_referring_sample(sample, dataset_type, split, idx)
        
        # Optionally verify image exists (skip for relative paths that will be resolved later)
        if check_images:
            image_path = processed_sample['images'][0]['image']
            if not os.path.exists(image_path):
                skipped += 1
                continue
        
        processed.append(processed_sample)
    
    print(f"  Processed {len(processed)} samples from {os.path.basename(file_path)}" + 
          (f" (skipped {skipped} missing images)" if skipped > 0 else ""))
    return processed


def main(
    local_dir: str = 'data/hoi/train_data_multitool',
    seed: int = 42,
    val_size: int = 500,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    include_hico: bool = True,
    include_swig: bool = True,
    check_images: bool = False,  # Default to False for relative paths
):
    """
    Prepare HOI dataset for RL training with MULTI-TOOL support and refer_boxes.
    
    Args:
        local_dir: Output directory for parquet files
        seed: Random seed for shuffling
        val_size: Number of samples for validation (from each test file)
        max_train_samples: Optional limit on training samples per file
        max_val_samples: Optional limit on validation samples per file
        include_hico: Whether to include HICO-DET data
        include_swig: Whether to include SWIG-HOI data
        check_images: Whether to verify image files exist (disable for relative paths)
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("HOI Dataset Preparation for RL Training (MULTI-TOOL + SPATIAL LINKING)")
    print("=" * 80)
    print(f"Output directory: {local_dir}")
    print(f"Random seed: {seed}")
    print(f"Include HICO: {include_hico}")
    print(f"Include SWIG: {include_swig}")
    print(f"Check images: {check_images}")
    print("=" * 80)
    print("Features:")
    print("  - Multi-tool system prompt (zoom_in, zoom_out, detect_objects)")
    print("  - refer_boxes column for spatial linking (3 boxes per sample)")
    print("  - Relative image paths for portability")
    print("=" * 80)
    
    # Collect training samples
    train_samples = []
    
    if include_hico:
        train_samples.extend(load_and_process_file(
            BENCHMARK_FILES['train']['hico_ground'], 'hico', 'ground', 'train', 
            max_train_samples, check_images
        ))
        train_samples.extend(load_and_process_file(
            BENCHMARK_FILES['train']['hico_referring'], 'hico', 'referring', 'train', 
            max_train_samples, check_images
        ))
    
    if include_swig:
        train_samples.extend(load_and_process_file(
            BENCHMARK_FILES['train']['swig_ground'], 'swig', 'ground', 'train', 
            max_train_samples, check_images
        ))
        train_samples.extend(load_and_process_file(
            BENCHMARK_FILES['train']['swig_referring'], 'swig', 'referring', 'train', 
            max_train_samples, check_images
        ))
    
    print(f"\nTotal training samples: {len(train_samples)}")
    
    # Collect validation samples (from test files)
    val_samples = []
    
    if include_hico:
        val_samples.extend(load_and_process_file(
            BENCHMARK_FILES['val']['hico_ground'], 'hico', 'ground', 'val', 
            max_val_samples or val_size, check_images
        ))
        val_samples.extend(load_and_process_file(
            BENCHMARK_FILES['val']['hico_referring'], 'hico', 'referring', 'val', 
            max_val_samples or val_size, check_images
        ))
    
    if include_swig:
        val_samples.extend(load_and_process_file(
            BENCHMARK_FILES['val']['swig_ground'], 'swig', 'ground', 'val', 
            max_val_samples or val_size, check_images
        ))
        val_samples.extend(load_and_process_file(
            BENCHMARK_FILES['val']['swig_referring'], 'swig', 'referring', 'val', 
            max_val_samples or val_size, check_images
        ))
    
    print(f"Total validation samples: {len(val_samples)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = datasets.Dataset.from_list(train_samples)
    val_dataset = datasets.Dataset.from_list(val_samples)
    
    # Shuffle training data
    train_dataset = train_dataset.shuffle(seed=seed)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    
    # Count by task type
    train_grounding = sum(1 for s in train_samples if s['ability'] == 'grounding')
    train_referring = sum(1 for s in train_samples if s['ability'] == 'referring')
    val_grounding = sum(1 for s in val_samples if s['ability'] == 'grounding')
    val_referring = sum(1 for s in val_samples if s['ability'] == 'referring')
    
    print(f"Training set:")
    print(f"  - Grounding: {train_grounding}")
    print(f"  - Referring: {train_referring}")
    print(f"  - Total: {len(train_samples)}")
    print(f"\nValidation set:")
    print(f"  - Grounding: {val_grounding}")
    print(f"  - Referring: {val_referring}")
    print(f"  - Total: {len(val_samples)}")
    
    # Print example
    print("\n" + "=" * 80)
    print("Example Training Sample")
    print("=" * 80)
    if train_samples:
        example = train_samples[0]
        print(f"Data source: {example['data_source']}")
        print(f"Ability: {example['ability']}")
        print(f"Image: {example['images'][0]['image']}")
        print(f"refer_boxes: {example['refer_boxes']}")
        print(f"System prompt (first 200 chars): {example['prompt'][0]['content'][:200]}...")
        
        # Verify multi-tool
        system_content = example['prompt'][0]['content']
        has_zoom_in = 'zoom_in' in system_content
        has_zoom_out = 'zoom_out' in system_content
        has_detect_objects = 'detect_objects' in system_content
        print(f"\nMulti-tool check:")
        print(f"  - zoom_in: {has_zoom_in}")
        print(f"  - zoom_out: {has_zoom_out}")
        print(f"  - detect_objects: {has_detect_objects}")
    
    # Save to parquet
    print("\n" + "=" * 80)
    print("Saving to Parquet")
    print("=" * 80)
    
    train_path = local_dir / 'train.parquet'
    val_path = local_dir / 'val.parquet'
    
    train_dataset.to_parquet(str(train_path))
    print(f"Saved {len(train_dataset)} training samples to {train_path}")
    
    val_dataset.to_parquet(str(val_path))
    print(f"Saved {len(val_dataset)} validation samples to {val_path}")
    
    # Save statistics
    stats = {
        "train": {
            "total": len(train_samples),
            "grounding": train_grounding,
            "referring": train_referring,
        },
        "val": {
            "total": len(val_samples),
            "grounding": val_grounding,
            "referring": val_referring,
        },
        "features": {
            "multi_tool": True,
            "refer_boxes": True,
            "relative_paths": True,
        }
    }
    stats_path = local_dir / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    print("\n" + "=" * 80)
    print("✓ Dataset preparation complete!")
    print("=" * 80)
    print(f"\nTo verify the dataset:")
    print(f"  python -c \"import pandas as pd; df = pd.read_parquet('{train_path}'); print(df.columns); print(df['refer_boxes'].iloc[0])\"")
    
    return train_dataset, val_dataset


if __name__ == '__main__':
    fire.Fire(main)
