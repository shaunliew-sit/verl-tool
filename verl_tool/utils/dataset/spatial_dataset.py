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
Spatial Linking Dataset Utilities for GRPO Training.

This module provides utilities for preparing and handling spatial linking data
in the verl-tool GRPO training pipeline.

Data Format:
    The training data (parquet) should include a 'refer_boxes' column with
    bounding boxes in the following format:
    
    {
        "prompt": [...],  # Standard prompt messages
        "images": ["path/to/image.jpg"],
        "refer_boxes": [
            [x1, y1, x2, y2],  # person box (0-1000 normalized)
            [x1, y1, x2, y2],  # object box
            [x1, y1, x2, y2],  # interaction box (optional, computed if missing)
        ],
        "reward_model": {"ground_truth": "riding bicycle"}
    }

Usage:
    from verl_tool.utils.dataset import prepare_spatial_linking_data
    
    # Prepare data with interaction boxes
    data = prepare_spatial_linking_data(raw_data)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_interaction_box(
    person_box: Union[List[float], torch.Tensor, np.ndarray],
    object_box: Union[List[float], torch.Tensor, np.ndarray],
) -> List[float]:
    """
    Compute the interaction box as the union of person and object boxes.
    
    The interaction box captures the spatial context where the interaction occurs,
    encompassing both entities and the space between them.
    
    Args:
        person_box: Person bounding box [x1, y1, x2, y2]
        object_box: Object bounding box [x1, y1, x2, y2]
        
    Returns:
        Interaction box [x1, y1, x2, y2] as list
    """
    # Convert to list if needed
    if isinstance(person_box, (torch.Tensor, np.ndarray)):
        person_box = person_box.tolist()
    if isinstance(object_box, (torch.Tensor, np.ndarray)):
        object_box = object_box.tolist()
    
    x1 = min(person_box[0], object_box[0])
    y1 = min(person_box[1], object_box[1])
    x2 = max(person_box[2], object_box[2])
    y2 = max(person_box[3], object_box[3])
    
    return [x1, y1, x2, y2]


def normalize_refer_boxes(
    boxes: Union[List[List[float]], torch.Tensor, np.ndarray],
    source_format: str = "1000",
    target_format: str = "0-1",
) -> List[List[float]]:
    """
    Normalize bounding boxes between different formats.
    
    Args:
        boxes: List of [x1, y1, x2, y2] boxes
        source_format: Source format ("1000" for 0-1000, "0-1" for normalized)
        target_format: Target format ("1000" for 0-1000, "0-1" for normalized)
        
    Returns:
        Normalized boxes
    """
    if isinstance(boxes, (torch.Tensor, np.ndarray)):
        boxes = boxes.tolist()
    
    if source_format == target_format:
        return boxes
    
    if source_format == "1000" and target_format == "0-1":
        return [[coord / 1000.0 for coord in box] for box in boxes]
    elif source_format == "0-1" and target_format == "1000":
        return [[coord * 1000.0 for coord in box] for box in boxes]
    else:
        raise ValueError(f"Unknown format conversion: {source_format} -> {target_format}")


def validate_refer_boxes(
    boxes: Optional[Union[List[List[float]], torch.Tensor, np.ndarray]],
    expected_count: int = 3,
    allow_none: bool = True,
) -> Tuple[bool, str]:
    """
    Validate refer_boxes format and content.
    
    Args:
        boxes: Bounding boxes to validate
        expected_count: Expected number of boxes (default 3: person, object, interaction)
        allow_none: Whether None is a valid value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if boxes is None:
        if allow_none:
            return True, ""
        return False, "refer_boxes is None but allow_none=False"
    
    if isinstance(boxes, (torch.Tensor, np.ndarray)):
        boxes = boxes.tolist()
    
    if not isinstance(boxes, list):
        return False, f"refer_boxes should be a list, got {type(boxes)}"
    
    if len(boxes) < expected_count:
        return False, f"Expected at least {expected_count} boxes, got {len(boxes)}"
    
    for i, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 4:
            return False, f"Box {i} should be [x1, y1, x2, y2], got {box}"
        
        # Check coordinates are valid
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2:
            return False, f"Box {i} has invalid coordinates: x1({x1}) >= x2({x2}) or y1({y1}) >= y2({y2})"
    
    return True, ""


def prepare_spatial_linking_data(
    data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
    add_interaction_box: bool = True,
    normalize_to: str = "1000",
    validate: bool = True,
) -> Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]:
    """
    Prepare spatial linking data for GRPO training.
    
    This function:
    1. Validates refer_boxes format
    2. Adds interaction box if missing (computed as union of person + object)
    3. Normalizes coordinates to the target format
    
    Args:
        data: Training data (dict, DataFrame, or list of dicts)
        add_interaction_box: Whether to add interaction box if missing
        normalize_to: Target coordinate format ("1000" or "0-1")
        validate: Whether to validate refer_boxes
        
    Returns:
        Processed data with valid refer_boxes
    """
    def process_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
        result = item.copy()
        
        if "refer_boxes" not in result or result["refer_boxes"] is None:
            return result
        
        boxes = result["refer_boxes"]
        
        # Convert to list if needed
        if isinstance(boxes, (torch.Tensor, np.ndarray)):
            boxes = boxes.tolist()
        
        # Validate
        if validate:
            is_valid, error_msg = validate_refer_boxes(boxes, expected_count=2, allow_none=True)
            if not is_valid:
                logger.warning(f"Invalid refer_boxes: {error_msg}. Skipping spatial linking for this sample.")
                result["refer_boxes"] = None
                return result
        
        # Detect source format
        if len(boxes) > 0 and len(boxes[0]) == 4:
            max_coord = max(max(box) for box in boxes)
            source_format = "1000" if max_coord > 1.0 else "0-1"
        else:
            source_format = normalize_to
        
        # Normalize
        if source_format != normalize_to:
            boxes = normalize_refer_boxes(boxes, source_format, normalize_to)
        
        # Add interaction box if needed
        if add_interaction_box and len(boxes) == 2:
            person_box = boxes[0]
            object_box = boxes[1]
            interaction_box = compute_interaction_box(person_box, object_box)
            
            # Convert back to target format if needed
            if normalize_to == "1000":
                interaction_box = [float(coord) for coord in interaction_box]
            
            boxes.append(interaction_box)
        
        result["refer_boxes"] = boxes
        return result
    
    # Handle different input types
    if isinstance(data, dict):
        return process_single_item(data)
    elif isinstance(data, pd.DataFrame):
        # Process each row
        processed = data.apply(lambda row: process_single_item(row.to_dict()), axis=1)
        return pd.DataFrame(list(processed))
    elif isinstance(data, list):
        return [process_single_item(item) for item in data]
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def create_spatial_parquet(
    input_data: List[Dict[str, Any]],
    output_path: str,
    add_interaction_box: bool = True,
    normalize_to: str = "1000",
) -> str:
    """
    Create a parquet file with spatial linking data for GRPO training.
    
    Args:
        input_data: List of training samples with refer_boxes
        output_path: Path to save the parquet file
        add_interaction_box: Whether to add interaction box if missing
        normalize_to: Target coordinate format
        
    Returns:
        Path to the created parquet file
    """
    # Process data
    processed_data = prepare_spatial_linking_data(
        input_data,
        add_interaction_box=add_interaction_box,
        normalize_to=normalize_to,
        validate=True,
    )
    
    # Create DataFrame and save
    df = pd.DataFrame(processed_data)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Created spatial linking parquet: {output_path} with {len(df)} samples")
    return output_path


def collate_refer_boxes(batch_boxes: List[Optional[List[List[float]]]]) -> List[Optional[torch.Tensor]]:
    """
    Collate refer_boxes from a batch into a list of tensors.
    
    This is used during training to convert refer_boxes from the dataset
    into the format expected by the spatial linking model.
    
    Args:
        batch_boxes: List of refer_boxes (one per sample in batch)
        
    Returns:
        List of tensors (or None) for each sample
    """
    result = []
    for boxes in batch_boxes:
        if boxes is None:
            result.append(None)
        else:
            # Convert to tensor
            if isinstance(boxes, torch.Tensor):
                result.append(boxes)
            elif isinstance(boxes, np.ndarray):
                result.append(torch.from_numpy(boxes).float())
            else:
                result.append(torch.tensor(boxes, dtype=torch.float32))
    return result
