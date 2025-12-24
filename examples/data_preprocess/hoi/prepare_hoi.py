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
Prepare HOI (Human-Object Interaction) dataset for RL training.

This script converts the simplified benchmark JSON files to parquet format
compatible with the verl-tool training pipeline.

Usage:
    python examples/data_preprocess/hoi/prepare_hoi.py \
        --local_dir data/hoi/train_data \
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
    - train.parquet: Combined training data
    - val.parquet: Validation data from test sets
"""

import fire
import os
import json
import datasets
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional

# System prompt with tool definitions (based on pixel_reasoner format)
SYSTEM_PROMPT = """You are a helpful assistant for Human-Object Interaction detection.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "zoom_in", "description": "Zoom in on a specific region of the image to examine details.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "Bounding box coordinates [x1, y1, x2, y2] in 1000x1000 normalized format.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to zoom in on. Use 1 for the main image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "zoom_out", "description": "Reset the view to the original full image.", "parameters": {"type": "object", "properties": {"target_image": {"type": "number", "description": "The index of the image to reset. Use 1 for the main image."}}, "required": ["target_image"]}}}
{"type": "function", "function": {"name": "detect_objects", "description": "Detect objects in the image using Grounding DINO.", "parameters": {"type": "object", "properties": {"class_names": {"type": "string", "description": "Object classes to detect, separated by ' . ' (e.g., 'person . cup . chair')."}, "target_image": {"type": "number", "description": "The index of the image to analyze. Use 1 for the main image."}, "confidence_threshold": {"type": "number", "description": "Minimum confidence for detections (0.0-1.0). Default: 0.25"}}, "required": ["class_names", "target_image"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

GROUNDING_GUIDELINE = """Guidelines: Analyze the image to locate human-object interaction pairs. You may use zoom_in to examine details or detect_objects to find candidates. For each person-object pair, output bounding boxes in JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "description"}, ...]. Coordinates should be in the 1000x1000 normalized format."""

REFERRING_GUIDELINE = """Guidelines: Analyze the provided bounding boxes to determine what action the person is performing with the object. You may use zoom_in to examine interaction details. Output only the action phrase (e.g., "riding bicycle", "sitting on bench"). Use base verb form without articles."""

# Image directory mapping
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
    Get the absolute image path based on dataset type and split.
    
    Args:
        file_name: Image filename
        dataset_type: 'hico' or 'swig'
        split: 'train' or 'val'
        
    Returns:
        Absolute path to the image
    """
    if dataset_type == 'hico':
        if split == 'train':
            base_dir = IMAGE_DIRS['hico_train']
        else:
            base_dir = IMAGE_DIRS['hico_test']
    else:  # swig
        base_dir = IMAGE_DIRS['swig']
    
    return str(Path(base_dir).absolute() / file_name)


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


def process_grounding_sample(sample: Dict[str, Any], dataset_type: str, split: str, idx: int) -> Dict[str, Any]:
    """
    Process a grounding sample into RL format.
    
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
    
    # Handle both formats
    if 'query' in sample:
        # Simplified format with pre-built query/response
        query = sample['query']
        response = sample['response']
    else:
        # Raw format - build query and response
        action = sample.get('action', 'interacting with')
        object_category = sample.get('object_category', 'object')
        num_pairs = sample.get('num_pairs', 1)
        
        query = f'Locate every person who is {action} {object_category} and the {object_category} they interact with in this image. For each person-object pair, output bbox coordinates in JSON format like: {{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}'
        
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
    
    # Build user content
    user_content = f"<image>{query}\n\n{GROUNDING_GUIDELINE}"
    
    # Keep ground truth as string (JSON) for consistency
    ground_truth = response if isinstance(response, str) else json.dumps(response)
    
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
    Process a referring sample into RL format.
    
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
    
    # Handle both formats
    if 'query' in sample:
        # Simplified format with pre-built query/response
        query = sample['query']
        response = sample['response']
    else:
        # Raw format - build query and response
        person_box_idx = sample.get('person_box_idx', 0)
        object_box_idx = sample.get('object_box_idx', 1)
        gt_action = sample.get('gt_action', '')
        
        # Get boxes
        person_box = boxes_1000[person_box_idx] if person_box_idx < len(boxes_1000) else [0, 0, 1000, 1000]
        object_box = boxes_1000[object_box_idx] if object_box_idx < len(boxes_1000) else [0, 0, 1000, 1000]
        
        query = f'Action Recognition Task: The first region {{"bbox_2d": {person_box}, "label": "person"}} contains a PERSON. The second region {{"bbox_2d": {object_box}, "label": "object"}} contains an OBJECT. Describe the action the person is performing with this object. Respond with only the action phrase (e.g., "riding bicycle", "sitting on bench").'
        response = gt_action
    
    # Build user content
    user_content = f"<image>{query}\n\n{REFERRING_GUIDELINE}"
    
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
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load a JSON file and process all samples.
    
    Args:
        file_path: Path to JSON file
        dataset_type: 'hico' or 'swig'
        task_type: 'ground' or 'referring'
        split: 'train' or 'val'
        max_samples: Optional limit on samples
        
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
    for idx, sample in enumerate(data):
        if task_type == 'ground':
            processed_sample = process_grounding_sample(sample, dataset_type, split, idx)
        else:
            processed_sample = process_referring_sample(sample, dataset_type, split, idx)
        
        # Verify image exists
        image_path = processed_sample['images'][0]['image']
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        processed.append(processed_sample)
    
    print(f"  Processed {len(processed)} samples from {os.path.basename(file_path)}")
    return processed


def main(
    local_dir: str = 'data/hoi/train_data',
    seed: int = 42,
    val_size: int = 500,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    include_hico: bool = True,
    include_swig: bool = True,
):
    """
    Prepare HOI dataset for RL training.
    
    Args:
        local_dir: Output directory for parquet files
        seed: Random seed for shuffling
        val_size: Number of samples for validation (from each test file)
        max_train_samples: Optional limit on training samples per file
        max_val_samples: Optional limit on validation samples per file
        include_hico: Whether to include HICO-DET data
        include_swig: Whether to include SWIG-HOI data
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("HOI Dataset Preparation for RL Training")
    print("=" * 80)
    print(f"Output directory: {local_dir}")
    print(f"Random seed: {seed}")
    print(f"Include HICO: {include_hico}")
    print(f"Include SWIG: {include_swig}")
    print("=" * 80)
    
    # Collect training samples
    train_samples = []
    
    if include_hico:
        train_samples.extend(load_and_process_file(
            BENCHMARK_FILES['train']['hico_ground'], 'hico', 'ground', 'train', max_train_samples
        ))
        train_samples.extend(load_and_process_file(
            BENCHMARK_FILES['train']['hico_referring'], 'hico', 'referring', 'train', max_train_samples
        ))
    
    if include_swig:
        train_samples.extend(load_and_process_file(
            BENCHMARK_FILES['train']['swig_ground'], 'swig', 'ground', 'train', max_train_samples
        ))
        train_samples.extend(load_and_process_file(
            BENCHMARK_FILES['train']['swig_referring'], 'swig', 'referring', 'train', max_train_samples
        ))
    
    print(f"\nTotal training samples: {len(train_samples)}")
    
    # Collect validation samples (from test files)
    val_samples = []
    
    if include_hico:
        val_samples.extend(load_and_process_file(
            BENCHMARK_FILES['val']['hico_ground'], 'hico', 'ground', 'val', max_val_samples or val_size
        ))
        val_samples.extend(load_and_process_file(
            BENCHMARK_FILES['val']['hico_referring'], 'hico', 'referring', 'val', max_val_samples or val_size
        ))
    
    if include_swig:
        val_samples.extend(load_and_process_file(
            BENCHMARK_FILES['val']['swig_ground'], 'swig', 'ground', 'val', max_val_samples or val_size
        ))
        val_samples.extend(load_and_process_file(
            BENCHMARK_FILES['val']['swig_referring'], 'swig', 'referring', 'val', max_val_samples or val_size
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
        print(f"System prompt: {example['prompt'][0]['content'][:200]}...")
        print(f"User query: {example['prompt'][1]['content'][:300]}...")
        print(f"Ground truth: {example['reward_model']['ground_truth']}")
    
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
    
    print("\n✓ Dataset preparation complete!")
    print(f"\nTo visualize the dataset:")
    print(f"  python view_parquet.py {train_path} --rows 5 --show-all")
    
    return train_dataset, val_dataset


if __name__ == '__main__':
    fire.Fire(main)

