#!/usr/bin/env python3
"""
Generate high-quality SFT dataset for HOI detection using teacher model (Qwen3-VL).

This script uses a teacher model via vLLM OpenAI-compatible API to generate
reasoning traces for HOI tasks, following the PixelReasoner methodology.

Key Features:
- Uses vLLM OpenAI API for fast inference
- Generates data in LLaMA-Factory ShareGPT format
- Quality-based filtering (keeps only high-scoring responses)
- Supports both referring and grounding tasks
- Incremental dataset building (10 → 100 → 500+ samples)

Usage:
    python generate_sft_with_teacher.py \
        --input data/benchmarks_simplified/hico_referring_train_simplified.json \
        --output data/hoi/sft_data/pilot_10.json \
        --api_base http://localhost:8000/v1 \
        --model_name Qwen/Qwen3-VL-4B-Instruct \
        --num_samples 10 \
        --quality_threshold 0.5
"""

import argparse
import base64
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

# System prompt adapted from PixelReasoner + HOI tools
SYSTEM_PROMPT_HOI = """You are a helpful assistant for Human-Object Interaction detection.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image_normalized", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "normalized coordinates for bounding box [x1, y1, x2, y2] in [0.0, 1.0] range", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "Index of image to crop (1 for main image)"}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "detect_objects", "description": "Detect objects using Grounding DINO", "parameters": {"type": "object", "properties": {"class_names": {"type": "string", "description": "Classes to detect, separated by ' . ' (e.g., 'person . bicycle')"}, "target_image": {"type": "number", "description": "Index of image (1 for main)"}, "confidence_threshold": {"type": "number", "description": "Minimum confidence (0.0-1.0)"}}, "required": ["class_names", "target_image"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

GUIDELINES_REFERRING = """Guidelines: Understand the given visual information. Determine if it is beneficial to employ the given visual operations (tools). You can look closer by `crop_image_normalized`. Reason with the visual information step by step, and provide the action phrase (e.g., "riding bicycle", "sitting on bench")."""

GUIDELINES_GROUNDING = """Guidelines: Understand the visual information. Determine if using `detect_objects` or `crop_image_normalized` would help. Reason step by step, and output bounding boxes in JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "person"}, ...]."""


def create_vllm_client(api_base: str = "http://localhost:8000/v1") -> OpenAI:
    """Create OpenAI client pointing to vLLM server."""
    return OpenAI(
        api_key="EMPTY",  # vLLM doesn't need API key
        base_url=api_base
    )


def image_to_base64_url(image_path: str) -> str:
    """Convert local image to base64 data URL for API."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Determine image format from extension
    ext = Path(image_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    else:
        mime_type = 'image/jpeg'  # Default

    return f"data:{mime_type};base64,{image_data}"


def load_hoi_benchmark(file_path: str) -> List[Dict]:
    """Load HOI benchmark data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {file_path}")
    return data


def select_challenging_samples(
    samples: List[Dict],
    max_samples: int,
    max_per_action: int = 20
) -> List[Dict]:
    """
    Select challenging samples for SFT dataset.

    Criteria:
    1. Diverse action categories (avoid over-representation)
    2. Balanced distribution
    """
    action_counts = defaultdict(int)
    selected = []

    # Shuffle for random selection
    import random
    random.shuffle(samples)

    for sample in samples:
        # Get action from ground_truth
        gt = sample.get('reward_model', {}).get('ground_truth', {})

        if sample.get('ability') == 'referring':
            action = gt.get('action', 'unknown')
        else:  # grounding
            action = gt.get('action', 'unknown')

        # Skip over-represented actions
        if action_counts[action] >= max_per_action:
            continue

        selected.append(sample)
        action_counts[action] += 1

        if len(selected) >= max_samples:
            break

    print(f"Selected {len(selected)} samples across {len(action_counts)} action categories")
    return selected


def create_query_for_sample(sample: Dict) -> str:
    """Create query text for a sample based on task type."""
    ability = sample.get('ability', 'referring')
    gt = sample.get('reward_model', {}).get('ground_truth', {})

    if ability == 'referring':
        # Referring task: given bboxes, predict action
        person_bbox = gt.get('person_bbox', [0.2, 0.3, 0.4, 0.5])
        object_bbox = gt.get('object_bbox', [0.5, 0.3, 0.7, 0.5])

        # Convert to normalized format if needed
        query = f"Look at the person at region [{person_bbox[0]:.3f}, {person_bbox[1]:.3f}, {person_bbox[2]:.3f}, {person_bbox[3]:.3f}] and the object at [{object_bbox[0]:.3f}, {object_bbox[1]:.3f}, {object_bbox[2]:.3f}, {object_bbox[3]:.3f}]. What interaction is happening?\n\n{GUIDELINES_REFERRING}"
    else:
        # Grounding task: given action, find all instances
        action = gt.get('action', 'unknown')
        object_name = gt.get('object', 'object')

        query = f"Locate every person who is {action} {object_name} and the {object_name} they interact with.\n\n{GUIDELINES_GROUNDING}"

    return query


def evaluate_response(
    response: str,
    ground_truth: Dict,
    task_type: str
) -> float:
    """
    Evaluate response quality against ground truth.

    Returns score between 0.0 and 1.0.
    """
    if task_type == 'referring':
        # Check if action is mentioned in response
        gt_action = ground_truth.get('action', '').lower()
        response_lower = response.lower()

        if gt_action in response_lower:
            return 1.0

        # Check for verb match
        gt_verb = gt_action.split()[0] if gt_action else ''
        if gt_verb and gt_verb in response_lower:
            return 0.5

        return 0.0

    else:  # grounding
        # Check if response contains bounding box format
        if 'bbox_2d' in response or '[' in response and ']' in response:
            return 0.8  # Assume format is correct if bboxes are present
        return 0.0


def generate_sft_example(
    sample: Dict,
    vllm_client: OpenAI,
    model_name: str,
    quality_threshold: float = 0.7
) -> Optional[Dict]:
    """
    Generate SFT example by calling vLLM OpenAI API.

    Args:
        sample: HOI sample with image_path, query, ground_truth, task_type
        vllm_client: OpenAI client pointing to vLLM
        model_name: Model name on vLLM server
        quality_threshold: Minimum quality score to keep example

    Returns:
        SFT example in ShareGPT format or None if quality too low
    """
    # 1. Get image path and create query
    image_path = sample.get('images', [{}])[0].get('image', '')
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None

    query = create_query_for_sample(sample)
    image_url = image_to_base64_url(image_path)

    # 2. Prepare messages (OpenAI format for vLLM)
    api_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_HOI
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": query}
            ]
        }
    ]

    # 3. Call vLLM API (fast inference)
    try:
        response = vllm_client.chat.completions.create(
            model=model_name,
            messages=api_messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {e}")
        return None

    # 4. Evaluate quality (compare to ground truth)
    gt = sample.get('reward_model', {}).get('ground_truth', {})
    ability = sample.get('ability', 'referring')
    score = evaluate_response(response_text, gt, ability)

    # 5. Only keep high-quality responses
    if score < quality_threshold:
        return None

    # 6. Format for LLaMA-Factory ShareGPT format (multimodal)
    sharegpt_example = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_HOI
            },
            {
                "role": "user",
                "content": f"<image>{query}"
            },
            {
                "role": "assistant",
                "content": response_text
            }
        ],
        "images": [image_path]  # Array of image paths
    }

    # Store metadata for analysis
    metadata = {
        "task_type": ability,
        "score": score,
        "action": gt.get('action', 'unknown')
    }

    return {
        "example": sharegpt_example,
        "metadata": metadata
    }


def save_sft_dataset(examples: List[Dict], output_path: str):
    """Save in JSON format for LLaMA-Factory ShareGPT."""
    # ShareGPT format expects a JSON array
    dataset = [ex["example"] for ex in examples]

    # Save as JSON array
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} examples to {output_path}")


def save_statistics(examples: List[Dict], output_path: str):
    """Save dataset statistics."""
    task_counts = defaultdict(int)
    action_counts = defaultdict(int)
    scores = []

    for ex in examples:
        metadata = ex["metadata"]
        task_counts[metadata["task_type"]] += 1
        action_counts[metadata["action"]] += 1
        scores.append(metadata["score"])

    stats = {
        "total_examples": len(examples),
        "task_distribution": dict(task_counts),
        "action_distribution": dict(action_counts),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0
    }

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to {output_path}")
    print(f"Average score: {stats['avg_score']:.3f}")
    print(f"Task distribution: {stats['task_distribution']}")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset with teacher model")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with HOI samples")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for SFT data")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="vLLM API base URL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Teacher model name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--quality_threshold", type=float, default=0.5, help="Minimum quality score (0.0-1.0)")
    parser.add_argument("--max_per_action", type=int, default=20, help="Max samples per action category")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark data
    print(f"\n=== Loading Data ===")
    samples = load_hoi_benchmark(args.input)

    # Select challenging samples
    print(f"\n=== Selecting Samples ===")
    selected = select_challenging_samples(samples, args.num_samples, args.max_per_action)

    # Create vLLM client
    print(f"\n=== Connecting to vLLM API ===")
    print(f"API Base: {args.api_base}")
    print(f"Model: {args.model_name}")
    vllm_client = create_vllm_client(args.api_base)

    # Generate examples
    print(f"\n=== Generating Examples ===")
    examples = []

    for sample in tqdm(selected, desc="Generating"):
        result = generate_sft_example(
            sample,
            vllm_client,
            args.model_name,
            args.quality_threshold
        )

        if result is not None:
            examples.append(result)

    print(f"\nGenerated {len(examples)} / {len(selected)} high-quality examples")
    print(f"Success rate: {len(examples) / len(selected) * 100:.1f}%")

    if len(examples) == 0:
        print("Warning: No examples generated. Check quality threshold or model responses.")
        return

    # Save dataset
    print(f"\n=== Saving Results ===")
    save_sft_dataset(examples, args.output)

    # Save statistics
    stats_path = str(Path(args.output).parent / "statistics.json")
    save_statistics(examples, stats_path)

    print(f"\n=== Done ===")
    print(f"Dataset: {args.output}")
    print(f"Statistics: {stats_path}")


if __name__ == "__main__":
    main()
