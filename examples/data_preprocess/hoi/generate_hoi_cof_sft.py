#!/usr/bin/env python3
"""
Generate Chain-of-Focus (CoF) style SFT dataset for HOI detection.

This script converts HICO-DET and SWIG-HOI benchmark data into CoF-style
multi-turn conversation format with actual cropped/zoomed images.

Output format matches exactly: data/CoF-SFT-Data-5.4k/cof_sft_data.json

Usage:
    python generate_hoi_cof_sft.py \
        --output_dir data/hoi_cof_sft \
        --max_samples 500 \
        --include_hico \
        --include_swig
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

# =============================================================================
# System Prompt for verl-tool compatibility (using zoom_in tool name)
# Coordinates use 1000-grid format for Qwen3-VL compatibility
# =============================================================================
SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"zoom_in","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d). Coordinates use 1000x1000 normalized format.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2] in 1000x1000 normalized format, where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"target_image":{"type":"number","description":"The index of the image to zoom in on. Use 1 for the main image."}},"required":["bbox_2d", "target_image"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# User instruction suffix (exact match from CoF)
USER_INSTRUCTION_SUFFIX = "Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."

# After tool call user message (exact match from CoF)
AFTER_TOOL_USER_MESSAGE = f"<image>\n{USER_INSTRUCTION_SUFFIX}"

# =============================================================================
# Data paths
# =============================================================================
BENCHMARK_FILES = {
    'hico_ground_train': 'data/benchmarks_simplified/hico_ground_train_simplified.json',
    'hico_referring_train': 'data/benchmarks_simplified/hico_referring_train_simplified.json',
    'swig_ground_train': 'data/benchmarks_simplified/swig_ground_train_simplified.json',
    'swig_referring_train': 'data/benchmarks_simplified/swig_referring_train_simplified.json',
}

IMAGE_DIRS = {
    'hico_train': 'data/hico_20160224_det/images/train2015',
    'swig': 'data/swig_hoi/images_512',
}


# =============================================================================
# Image Processing Functions
# =============================================================================

def compute_interaction_bbox(
    person_box: List[int],
    object_box: List[int],
    img_width: int,
    img_height: int,
    padding: float = 0.15
) -> List[int]:
    """
    Compute the union bounding box of person and object with padding.
    
    Args:
        person_box: [x1, y1, x2, y2] in pixels
        object_box: [x1, y1, x2, y2] in pixels
        img_width: Image width
        img_height: Image height
        padding: Padding ratio (0.15 = 15% of bbox dimensions)
    
    Returns:
        Union bbox with padding [x1, y1, x2, y2]
    """
    # Compute union bbox
    x1 = min(person_box[0], object_box[0])
    y1 = min(person_box[1], object_box[1])
    x2 = max(person_box[2], object_box[2])
    y2 = max(person_box[3], object_box[3])
    
    # Add padding
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_width, x2 + pad_x)
    y2 = min(img_height, y2 + pad_y)
    
    return [x1, y1, x2, y2]


def compute_bbox_area_ratio(bbox: List[int], img_width: int, img_height: int) -> float:
    """Compute the ratio of bbox area to image area."""
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    img_area = img_width * img_height
    return bbox_area / img_area if img_area > 0 else 0


def crop_and_save_images(
    src_image_path: str,
    output_dir: str,
    sample_id: int,
    zoom_bbox: Optional[List[int]] = None,
    scale_factor: float = 2.0
) -> Tuple[str, Optional[str]]:
    """
    Copy original image and create cropped zoom image.
    
    Args:
        src_image_path: Path to source image
        output_dir: Output directory for images
        sample_id: Sample ID for folder naming
        zoom_bbox: Bbox to crop [x1, y1, x2, y2], None if no zoom needed
        scale_factor: Scale factor for zoomed image (default 2x)
    
    Returns:
        Tuple of (original_relative_path, zoomed_relative_path or None)
    """
    # Create sample directory
    sample_dir = os.path.join(output_dir, "images", str(sample_id))
    os.makedirs(sample_dir, exist_ok=True)
    
    # Copy/save original image
    original_path = os.path.join(sample_dir, "0.jpg")
    img = Image.open(src_image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img.save(original_path, 'JPEG', quality=95)
    
    relative_original = f"{sample_id}/0.jpg"
    
    if zoom_bbox is None:
        return relative_original, None
    
    # Crop and zoom
    x1, y1, x2, y2 = zoom_bbox
    cropped = img.crop((x1, y1, x2, y2))
    
    # Scale up by factor
    new_width = int(cropped.width * scale_factor)
    new_height = int(cropped.height * scale_factor)
    if new_width > 0 and new_height > 0:
        cropped = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save zoomed image
    zoomed_path = os.path.join(sample_dir, "1.jpg")
    cropped.save(zoomed_path, 'JPEG', quality=95)
    
    relative_zoomed = f"{sample_id}/1.jpg"
    
    return relative_original, relative_zoomed


def convert_boxes_1000_to_pixels(boxes_1000: List[List[int]], width: int, height: int) -> List[List[int]]:
    """Convert boxes from 1000x1000 normalized format to pixel coordinates with clamping."""
    result = []
    for box in boxes_1000:
        x1 = max(0, min(int(box[0] / 1000 * width), width - 1))
        y1 = max(0, min(int(box[1] / 1000 * height), height - 1))
        x2 = max(0, min(int(box[2] / 1000 * width), width - 1))
        y2 = max(0, min(int(box[3] / 1000 * height), height - 1))
        result.append([x1, y1, x2, y2])
    return result


def clamp_boxes_1000(boxes_1000: List[List[int]]) -> List[List[int]]:
    """Clamp 1000-grid coordinates to valid range [0, 999]."""
    result = []
    for box in boxes_1000:
        x1 = max(0, min(int(box[0]), 999))
        y1 = max(0, min(int(box[1]), 999))
        x2 = max(0, min(int(box[2]), 999))
        y2 = max(0, min(int(box[3]), 999))
        result.append([x1, y1, x2, y2])
    return result


def compute_interaction_bbox_1000(
    person_box_1000: List[int],
    object_box_1000: List[int],
    padding: float = 0.15
) -> List[int]:
    """
    Compute the union bounding box of person and object with padding in 1000-grid format.
    
    Args:
        person_box_1000: [x1, y1, x2, y2] in 1000-grid format
        object_box_1000: [x1, y1, x2, y2] in 1000-grid format
        padding: Padding ratio (0.15 = 15% of bbox dimensions)
    
    Returns:
        Union bbox with padding [x1, y1, x2, y2] in 1000-grid format
    """
    # Compute union bbox
    x1 = min(person_box_1000[0], object_box_1000[0])
    y1 = min(person_box_1000[1], object_box_1000[1])
    x2 = max(person_box_1000[2], object_box_1000[2])
    y2 = max(person_box_1000[3], object_box_1000[3])
    
    # Add padding
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(999, x2 + pad_x)
    y2 = min(999, y2 + pad_y)
    
    return [x1, y1, x2, y2]


# =============================================================================
# Conversation Generation - Referring Task
# =============================================================================

def generate_referring_zoom_conversation(
    person_box_1000: List[int],
    object_box_1000: List[int],
    zoom_bbox_1000: List[int],
    action: str,
    object_category: str
) -> List[Dict[str, str]]:
    """
    Generate multi-turn conversation for referring task with zoom.
    
    All coordinates are in 1000-grid format for Qwen3-VL compatibility.
    Output format: "{verb} {object}" e.g., "riding motorcycle"
    """
    # Format answer: verb + object
    answer = f"{action} {object_category}"
    
    # User message (turn 1) - coordinates in 1000-grid format
    user_content = f"""<image> Question: What action is the person performing with the object?
The person is located at {person_box_1000} and the object is at {object_box_1000}.
Respond with ONLY the action phrase in format: "{{verb}} {{object}}" (e.g., "riding bicycle", "holding cup"). Use base verb form, no articles.
{USER_INSTRUCTION_SUFFIX}"""

    # Assistant reasoning for zoom - using zoom_in tool with target_image
    think_zoom = f"To identify the action between the person and object, I need to examine the interaction region more closely. The person is at {person_box_1000} and the object is at {object_box_1000}. I will zoom in on the interaction area to see the details of how they are interacting."
    
    assistant_zoom = f"""<think> {think_zoom} </think>
<tool_call>
{{"name": "zoom_in", "arguments": {{"bbox_2d": {zoom_bbox_1000}, "target_image": 1}}}}
</tool_call>"""

    # Final answer after zoom
    think_final = f"Now I can clearly see the interaction between the person and the {object_category}. Based on the person's posture and position relative to the {object_category}, the action is {action}."
    
    assistant_final = f"""<think> {think_final} </think>
<answer> {answer} </answer>"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_zoom},
        {"role": "user", "content": AFTER_TOOL_USER_MESSAGE},
        {"role": "assistant", "content": assistant_final},
    ]


def generate_referring_direct_conversation(
    person_box_1000: List[int],
    object_box_1000: List[int],
    action: str,
    object_category: str
) -> List[Dict[str, str]]:
    """
    Generate single-turn conversation for referring task (no zoom needed).
    
    All coordinates are in 1000-grid format for Qwen3-VL compatibility.
    Output format: "{verb} {object}" e.g., "riding motorcycle"
    """
    # Format answer: verb + object
    answer = f"{action} {object_category}"
    
    # User message - coordinates in 1000-grid format
    user_content = f"""<image> Question: What action is the person performing with the object?
The person is located at {person_box_1000} and the object is at {object_box_1000}.
Respond with ONLY the action phrase in format: "{{verb}} {{object}}" (e.g., "riding bicycle", "holding cup"). Use base verb form, no articles.
{USER_INSTRUCTION_SUFFIX}"""

    # Direct answer
    think = f"The interaction region is clearly visible. The person is at {person_box_1000} and the {object_category} is at {object_box_1000}. Based on the person's posture and position, they are {action} the {object_category}."
    
    assistant_content = f"""<think> {think} </think>
<answer> {answer} </answer>"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# =============================================================================
# Conversation Generation - Grounding Task
# =============================================================================

def generate_grounding_zoom_conversation(
    boxes_1000: List[List[int]],
    zoom_bbox_1000: List[int],
    action: str,
    object_category: str,
    num_pairs: int
) -> List[Dict[str, str]]:
    """
    Generate multi-turn conversation for grounding task with zoom.
    
    All coordinates are in 1000-grid format for Qwen3-VL compatibility.
    Output format: JSON array of bbox objects
    """
    # Build answer JSON with 1000-grid coordinates
    answer_objects = []
    for i in range(0, min(len(boxes_1000), num_pairs * 2), 2):
        if i < len(boxes_1000):
            answer_objects.append({"bbox_2d": boxes_1000[i], "label": "person"})
        if i + 1 < len(boxes_1000):
            answer_objects.append({"bbox_2d": boxes_1000[i + 1], "label": object_category})
    
    answer_json = json.dumps(answer_objects)
    
    # User message (turn 1) - note about 1000-grid format
    user_content = f"""<image> Question: Locate every person who is {action} {object_category} and the {object_category} they interact with.
For each person-object pair, output bbox coordinates in JSON format like: {{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}. Coordinates should be in 1000x1000 normalized format.
{USER_INSTRUCTION_SUFFIX}"""

    # Assistant reasoning for zoom - using zoom_in tool with target_image
    think_zoom = f"To accurately locate all persons who are {action} {object_category}, I need to examine the image more closely. I will zoom in on the region where the interactions are occurring to identify all person-object pairs."
    
    assistant_zoom = f"""<think> {think_zoom} </think>
<tool_call>
{{"name": "zoom_in", "arguments": {{"bbox_2d": {zoom_bbox_1000}, "target_image": 1}}}}
</tool_call>"""

    # Final answer after zoom
    if num_pairs == 1:
        think_final = f"Now I can clearly identify the person and the {object_category} they are interacting with. The person is {action} the {object_category}."
    else:
        think_final = f"Now I can clearly identify {num_pairs} person-{object_category} pairs in the image. Each person is {action} a {object_category}."
    
    assistant_final = f"""<think> {think_final} </think>
<answer> {answer_json} </answer>"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_zoom},
        {"role": "user", "content": AFTER_TOOL_USER_MESSAGE},
        {"role": "assistant", "content": assistant_final},
    ]


def generate_grounding_direct_conversation(
    boxes_1000: List[List[int]],
    action: str,
    object_category: str,
    num_pairs: int
) -> List[Dict[str, str]]:
    """
    Generate single-turn conversation for grounding task (no zoom needed).
    
    All coordinates are in 1000-grid format for Qwen3-VL compatibility.
    Output format: JSON array of bbox objects
    """
    # Build answer JSON with 1000-grid coordinates
    answer_objects = []
    for i in range(0, min(len(boxes_1000), num_pairs * 2), 2):
        if i < len(boxes_1000):
            answer_objects.append({"bbox_2d": boxes_1000[i], "label": "person"})
        if i + 1 < len(boxes_1000):
            answer_objects.append({"bbox_2d": boxes_1000[i + 1], "label": object_category})
    
    answer_json = json.dumps(answer_objects)
    
    # User message - note about 1000-grid format
    user_content = f"""<image> Question: Locate every person who is {action} {object_category} and the {object_category} they interact with.
For each person-object pair, output bbox coordinates in JSON format like: {{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}. Coordinates should be in 1000x1000 normalized format.
{USER_INSTRUCTION_SUFFIX}"""

    # Direct answer
    if num_pairs == 1:
        think = f"The person and the {object_category} are clearly visible in the image. The person is {action} the {object_category}."
    else:
        think = f"I can clearly identify {num_pairs} person-{object_category} pairs in the image. Each person is {action} a {object_category}."
    
    assistant_content = f"""<think> {think} </think>
<answer> {answer_json} </answer>"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# =============================================================================
# Data Loading
# =============================================================================

def load_benchmark_file(file_path: str) -> List[Dict]:
    """Load benchmark JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {os.path.basename(file_path)}")
    return data


def get_image_path(file_name: str, dataset: str) -> str:
    """Get full image path based on dataset type."""
    if dataset == 'hico':
        base_dir = IMAGE_DIRS['hico_train']
    else:  # swig
        base_dir = IMAGE_DIRS['swig']
    
    return os.path.join(base_dir, file_name)


# =============================================================================
# Main Processing
# =============================================================================

def process_referring_sample(
    sample: Dict,
    dataset: str,
    output_dir: str,
    sample_id: int,
    zoom_threshold: float,
    padding: float
) -> Optional[Dict]:
    """Process a referring task sample.
    
    Uses 1000-grid coordinates for conversations (Qwen3-VL compatible).
    Uses pixel coordinates for image cropping.
    """
    file_name = sample.get('file_name', '')
    if not file_name:
        return None
    
    image_path = get_image_path(file_name, dataset)
    if not os.path.exists(image_path):
        return None
    
    # Get image dimensions
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Get boxes in 1000-grid format (used for conversation)
    boxes_1000 = sample.get('boxes_1000', [])
    if len(boxes_1000) < 2:
        return None
    
    # Clamp 1000-grid coordinates
    boxes_1000 = clamp_boxes_1000(boxes_1000)
    
    # Convert to pixel coordinates for image cropping
    boxes_pixel = convert_boxes_1000_to_pixels(boxes_1000, width, height)
    
    person_idx = sample.get('person_box_idx', 0)
    object_idx = sample.get('object_box_idx', 1)
    
    if person_idx >= len(boxes_1000) or object_idx >= len(boxes_1000):
        return None
    
    # 1000-grid boxes for conversation
    person_box_1000 = boxes_1000[person_idx]
    object_box_1000 = boxes_1000[object_idx]
    
    # Pixel boxes for image cropping
    person_box_pixel = boxes_pixel[person_idx]
    object_box_pixel = boxes_pixel[object_idx]
    
    # Get action and object from response
    response = sample.get('response', '')
    if not response:
        return None
    
    # Parse action and object from response
    # Response format is like "racing motorcycle" or "riding bicycle"
    parts = response.strip().split()
    if len(parts) >= 2:
        action = parts[0]
        object_category = ' '.join(parts[1:])
    else:
        action = response.strip()
        object_category = sample.get('object_category', 'object')
    
    # Compute interaction bbox in 1000-grid format (for conversation)
    interaction_bbox_1000 = compute_interaction_bbox_1000(person_box_1000, object_box_1000, padding)
    
    # Compute interaction bbox in pixel format (for image cropping)
    interaction_bbox_pixel = compute_interaction_bbox(person_box_pixel, object_box_pixel, width, height, padding)
    
    # Decide zoom or direct based on pixel area ratio
    area_ratio = compute_bbox_area_ratio(interaction_bbox_pixel, width, height)
    needs_zoom = area_ratio <= zoom_threshold
    
    # Generate images (using pixel coordinates for cropping)
    if needs_zoom:
        orig_path, zoom_path = crop_and_save_images(
            image_path, output_dir, sample_id, interaction_bbox_pixel
        )
        images = [orig_path, zoom_path]
        # Use 1000-grid coordinates in conversation
        messages = generate_referring_zoom_conversation(
            person_box_1000, object_box_1000, interaction_bbox_1000, action, object_category
        )
    else:
        orig_path, _ = crop_and_save_images(
            image_path, output_dir, sample_id, None
        )
        images = [orig_path]
        # Use 1000-grid coordinates in conversation
        messages = generate_referring_direct_conversation(
            person_box_1000, object_box_1000, action, object_category
        )
    
    return {
        "messages": messages,
        "images": images,
    }


def process_grounding_sample(
    sample: Dict,
    dataset: str,
    output_dir: str,
    sample_id: int,
    zoom_threshold: float,
    padding: float
) -> Optional[Dict]:
    """Process a grounding task sample.
    
    Uses 1000-grid coordinates for conversations (Qwen3-VL compatible).
    Uses pixel coordinates for image cropping.
    """
    file_name = sample.get('file_name', '')
    if not file_name:
        return None
    
    image_path = get_image_path(file_name, dataset)
    if not os.path.exists(image_path):
        return None
    
    # Get image dimensions
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Get boxes in 1000-grid format (used for conversation)
    boxes_1000 = sample.get('boxes_1000', [])
    if len(boxes_1000) < 2:
        return None
    
    # Clamp 1000-grid coordinates
    boxes_1000 = clamp_boxes_1000(boxes_1000)
    
    # Convert to pixel coordinates for image cropping
    boxes_pixel = convert_boxes_1000_to_pixels(boxes_1000, width, height)
    
    action = sample.get('action', '')
    object_category = sample.get('object_category', '')
    num_pairs = sample.get('num_pairs', 1)
    
    if not action or not object_category:
        return None
    
    # Compute overall interaction bbox in 1000-grid format (for conversation)
    all_x1_1000 = min(box[0] for box in boxes_1000)
    all_y1_1000 = min(box[1] for box in boxes_1000)
    all_x2_1000 = max(box[2] for box in boxes_1000)
    all_y2_1000 = max(box[3] for box in boxes_1000)
    
    # Add padding in 1000-grid
    w_1000 = all_x2_1000 - all_x1_1000
    h_1000 = all_y2_1000 - all_y1_1000
    pad_x_1000 = int(w_1000 * padding)
    pad_y_1000 = int(h_1000 * padding)
    
    interaction_bbox_1000 = [
        max(0, all_x1_1000 - pad_x_1000),
        max(0, all_y1_1000 - pad_y_1000),
        min(999, all_x2_1000 + pad_x_1000),
        min(999, all_y2_1000 + pad_y_1000)
    ]
    
    # Compute overall interaction bbox in pixel format (for image cropping)
    all_x1_pixel = min(box[0] for box in boxes_pixel)
    all_y1_pixel = min(box[1] for box in boxes_pixel)
    all_x2_pixel = max(box[2] for box in boxes_pixel)
    all_y2_pixel = max(box[3] for box in boxes_pixel)
    
    w_pixel = all_x2_pixel - all_x1_pixel
    h_pixel = all_y2_pixel - all_y1_pixel
    pad_x_pixel = int(w_pixel * padding)
    pad_y_pixel = int(h_pixel * padding)
    
    interaction_bbox_pixel = [
        max(0, all_x1_pixel - pad_x_pixel),
        max(0, all_y1_pixel - pad_y_pixel),
        min(width - 1, all_x2_pixel + pad_x_pixel),
        min(height - 1, all_y2_pixel + pad_y_pixel)
    ]
    
    # Decide zoom or direct based on pixel area ratio
    area_ratio = compute_bbox_area_ratio(interaction_bbox_pixel, width, height)
    needs_zoom = area_ratio <= zoom_threshold
    
    # Generate images (using pixel coordinates for cropping)
    if needs_zoom:
        orig_path, zoom_path = crop_and_save_images(
            image_path, output_dir, sample_id, interaction_bbox_pixel
        )
        images = [orig_path, zoom_path]
        # Use 1000-grid coordinates in conversation
        messages = generate_grounding_zoom_conversation(
            boxes_1000, interaction_bbox_1000, action, object_category, num_pairs
        )
    else:
        orig_path, _ = crop_and_save_images(
            image_path, output_dir, sample_id, None
        )
        images = [orig_path]
        # Use 1000-grid coordinates in conversation
        messages = generate_grounding_direct_conversation(
            boxes_1000, action, object_category, num_pairs
        )
    
    return {
        "messages": messages,
        "images": images,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HOI CoF-style SFT dataset")
    parser.add_argument("--output_dir", type=str, default="data/hoi_cof_sft",
                        help="Output directory for dataset")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum total samples to generate")
    parser.add_argument("--zoom_threshold", type=float, default=0.15,
                        help="Area ratio threshold for zoom decision (default: 0.15)")
    parser.add_argument("--padding", type=float, default=0.15,
                        help="Bbox padding ratio (default: 0.15)")
    parser.add_argument("--include_hico", action="store_true", default=True,
                        help="Include HICO-DET data")
    parser.add_argument("--include_swig", action="store_true", default=True,
                        help="Include SWIG-HOI data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_per_action", type=int, default=50,
                        help="Maximum samples per action type for diversity")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    
    print("=" * 80)
    print("HOI Chain-of-Focus SFT Dataset Generator")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Zoom threshold: {args.zoom_threshold}")
    print(f"Padding: {args.padding}")
    print("=" * 80)
    
    # Load all samples
    all_samples = []
    
    if args.include_hico:
        # HICO referring
        hico_referring = load_benchmark_file(BENCHMARK_FILES['hico_referring_train'])
        for s in hico_referring:
            s['_dataset'] = 'hico'
            s['_task'] = 'referring'
        all_samples.extend(hico_referring)
        
        # HICO grounding
        hico_ground = load_benchmark_file(BENCHMARK_FILES['hico_ground_train'])
        for s in hico_ground:
            s['_dataset'] = 'hico'
            s['_task'] = 'grounding'
        all_samples.extend(hico_ground)
    
    if args.include_swig:
        # SWIG referring
        swig_referring = load_benchmark_file(BENCHMARK_FILES['swig_referring_train'])
        for s in swig_referring:
            s['_dataset'] = 'swig'
            s['_task'] = 'referring'
        all_samples.extend(swig_referring)
        
        # SWIG grounding
        swig_ground = load_benchmark_file(BENCHMARK_FILES['swig_ground_train'])
        for s in swig_ground:
            s['_dataset'] = 'swig'
            s['_task'] = 'grounding'
        all_samples.extend(swig_ground)
    
    print(f"\nTotal samples loaded: {len(all_samples)}")
    
    # Shuffle and select diverse samples
    random.shuffle(all_samples)
    
    # Limit by action type for diversity
    action_counts = defaultdict(int)
    selected_samples = []
    
    for sample in all_samples:
        action = sample.get('action', sample.get('response', 'unknown'))
        if action_counts[action] < args.max_per_action:
            selected_samples.append(sample)
            action_counts[action] += 1
        
        if len(selected_samples) >= args.max_samples * 2:  # Get extra for filtering failures
            break
    
    print(f"Selected {len(selected_samples)} diverse samples")
    
    # Process samples
    print("\n" + "=" * 80)
    print("Processing samples...")
    print("=" * 80)
    
    dataset_samples = []
    sample_id = 0
    
    stats = {
        'total_processed': 0,
        'referring_zoom': 0,
        'referring_direct': 0,
        'grounding_zoom': 0,
        'grounding_direct': 0,
        'failed': 0,
    }
    
    for sample in tqdm(selected_samples, desc="Generating"):
        if len(dataset_samples) >= args.max_samples:
            break
        
        dataset = sample['_dataset']
        task = sample['_task']
        
        try:
            if task == 'referring':
                result = process_referring_sample(
                    sample, dataset, str(output_dir), sample_id,
                    args.zoom_threshold, args.padding
                )
                if result:
                    dataset_samples.append(result)
                    sample_id += 1
                    if len(result['images']) > 1:
                        stats['referring_zoom'] += 1
                    else:
                        stats['referring_direct'] += 1
                else:
                    stats['failed'] += 1
            else:  # grounding
                result = process_grounding_sample(
                    sample, dataset, str(output_dir), sample_id,
                    args.zoom_threshold, args.padding
                )
                if result:
                    dataset_samples.append(result)
                    sample_id += 1
                    if len(result['images']) > 1:
                        stats['grounding_zoom'] += 1
                    else:
                        stats['grounding_direct'] += 1
                else:
                    stats['failed'] += 1
        except Exception as e:
            print(f"Error processing sample: {e}")
            stats['failed'] += 1
    
    stats['total_processed'] = len(dataset_samples)
    
    # Save dataset
    print("\n" + "=" * 80)
    print("Saving dataset...")
    print("=" * 80)
    
    output_file = output_dir / "hoi_cof_sft_data.json"
    with open(output_file, 'w') as f:
        json.dump(dataset_samples, f, indent=4)
    
    print(f"Saved {len(dataset_samples)} samples to {output_file}")
    
    # Save statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved statistics to {stats_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total samples generated: {stats['total_processed']}")
    print(f"  - Referring (zoom): {stats['referring_zoom']}")
    print(f"  - Referring (direct): {stats['referring_direct']}")
    print(f"  - Grounding (zoom): {stats['grounding_zoom']}")
    print(f"  - Grounding (direct): {stats['grounding_direct']}")
    print(f"  - Failed: {stats['failed']}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Dataset file: {output_file}")
    
    # Print example
    if dataset_samples:
        print("\n" + "=" * 80)
        print("Example Sample")
        print("=" * 80)
        example = dataset_samples[0]
        print(f"Images: {example['images']}")
        print(f"Messages ({len(example['messages'])} turns):")
        for i, msg in enumerate(example['messages']):
            role = msg['role']
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            print(f"  [{i}] {role}: {content}")


if __name__ == "__main__":
    main()
