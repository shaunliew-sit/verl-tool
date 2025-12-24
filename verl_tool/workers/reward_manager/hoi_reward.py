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
HOI Reward Manager for Human-Object Interaction Detection.

VTool-style outcome-based rewards:
- Grounding: Binary IoU >= 0.5 reward
- Referring: Hybrid BERTScore (exact match -> 1.0, else BERTScore F1)

Uses DeBERTa-v2-xxlarge-mnli for BERTScore computation.
"""

import os
import torch
import json
import regex as re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union

from verl.workers.reward_manager import register
from verl import DataProto

import logging

logger = logging.getLogger(__name__)

# Global cache for BERTScore model
_bertscore_model = None
_bertscore_device = None


def get_bertscore_model(device: str = None):
    """
    Lazily load BERTScore model (cached after first load).
    Uses microsoft/deberta-v2-xxlarge-mnli for best performance.
    """
    global _bertscore_model, _bertscore_device
    
    if _bertscore_model is not None:
        return _bertscore_model
    
    try:
        from bert_score import BERTScorer
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _bertscore_device = device
        
        logger.info("Loading BERTScore model: microsoft/deberta-v2-xxlarge-mnli")
        _bertscore_model = BERTScorer(
            model_type="microsoft/deberta-v2-xxlarge-mnli",
            lang="en",
            rescale_with_baseline=True,
            device=device
        )
        logger.info(f"BERTScore model loaded successfully on {device}")
        return _bertscore_model
        
    except ImportError as e:
        logger.warning(f"BERTScore not available: {e}")
        logger.warning("Install with: pip install bert_score")
        return None
    except Exception as e:
        logger.warning(f"Failed to load BERTScore model: {e}")
        return None


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def clean_text(text: str) -> str:
    """
    Clean text for comparison.
    
    Handles:
    - Markdown formatting
    - Extra whitespace
    - Case normalization
    """
    if text is None:
        return ""
    
    text = str(text)
    
    # Remove markdown bold: **text** -> text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # Remove markdown italic: *text* -> text
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\1', text)
    
    # Remove markdown headers: # Header -> Header
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip().lower()


def extract_boxes_from_response(response_str: str) -> List[List[float]]:
    """
    Extract bounding boxes from model response.
    
    Handles formats like:
    - [{"bbox_2d": [x1, y1, x2, y2], "label": "person"}, ...]
    - [[x1, y1, x2, y2], ...]
    - [x1, y1, x2, y2]
    
    Returns:
        List of bounding boxes, each as [x1, y1, x2, y2]
    """
    boxes = []
    
    # Try to parse as JSON first
    try:
        # Look for JSON array in the response
        json_match = re.search(r'\[.*\]', response_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'bbox_2d' in item:
                        box = item['bbox_2d']
                        if isinstance(box, list) and len(box) == 4:
                            boxes.append([float(x) for x in box])
                    elif isinstance(item, list) and len(item) == 4:
                        boxes.append([float(x) for x in item])
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: use regex to find coordinate patterns
    if not boxes:
        # Match patterns like [x1, y1, x2, y2] or (x1, y1, x2, y2)
        pattern = r'[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]'
        matches = re.findall(pattern, response_str)
        for m in matches:
            boxes.append([float(m[0]), float(m[1]), float(m[2]), float(m[3])])
    
    return boxes


def grounding_score(response_str: str, ground_truth_boxes: List[List[float]], iou_threshold: float = 0.5) -> float:
    """
    Compute grounding score using binary IoU threshold.
    
    Args:
        response_str: Model response containing predicted bounding boxes
        ground_truth_boxes: List of ground truth boxes
        iou_threshold: IoU threshold for considering a match (default: 0.5)
        
    Returns:
        1.0 if all GT boxes are matched with IoU >= threshold, else 0.0
    """
    pred_boxes = extract_boxes_from_response(response_str)
    
    if not pred_boxes or not ground_truth_boxes:
        return 0.0
    
    # For each GT box, check if there's a matching prediction
    matched_gt = 0
    used_preds = set()
    
    for gt_box in ground_truth_boxes:
        best_iou = 0.0
        best_pred_idx = -1
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            if pred_idx in used_preds:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        
        if best_iou >= iou_threshold and best_pred_idx >= 0:
            matched_gt += 1
            used_preds.add(best_pred_idx)
    
    # Binary reward: 1.0 if all GT boxes matched, else 0.0
    if matched_gt == len(ground_truth_boxes):
        return 1.0
    else:
        return 0.0


def referring_score(response_str: str, ground_truth: str, bertscore_model=None) -> float:
    """
    Compute referring score using hybrid approach.
    
    Args:
        response_str: Model response containing action phrase
        ground_truth: Ground truth action phrase
        bertscore_model: BERTScorer instance (optional, will load if needed)
        
    Returns:
        1.0 for exact match, else BERTScore F1
    """
    # Clean and normalize text
    pred_clean = clean_text(response_str)
    gt_clean = clean_text(ground_truth)
    
    # Exact match check
    if pred_clean == gt_clean:
        return 1.0
    
    # Extract action phrase from response (may contain reasoning)
    # Look for patterns like "ACTION: ..." or just the last meaningful line
    action_match = re.search(r'\*?\*?action\*?\*?[:\s]+(.+?)(?:\n|$)', pred_clean, re.IGNORECASE)
    if action_match:
        pred_action = clean_text(action_match.group(1))
        if pred_action == gt_clean:
            return 1.0
    else:
        # Use last non-empty line as the action
        lines = [l.strip() for l in pred_clean.split('\n') if l.strip()]
        if lines:
            pred_action = lines[-1]
            if pred_action == gt_clean:
                return 1.0
    
    # Use BERTScore F1 for semantic similarity
    if bertscore_model is None:
        bertscore_model = get_bertscore_model()
    
    if bertscore_model is None:
        # Fallback: simple word overlap if BERTScore unavailable
        pred_words = set(pred_clean.split())
        gt_words = set(gt_clean.split())
        if not gt_words:
            return 0.0
        overlap = len(pred_words & gt_words) / len(gt_words)
        return overlap
    
    try:
        P, R, F1 = bertscore_model.score([pred_clean], [gt_clean])
        return F1[0].item()
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return 0.0


@register("hoi_reward")
class HOIRewardManager:
    """
    HOI Reward Manager with VTool-style outcome-based rewards.
    
    - Grounding: Binary IoU >= 0.5 reward
    - Referring: Hybrid (exact match -> 1.0, else BERTScore F1)
    
    No intermediate penalties (curiosity, redundancy) - pure outcome-based.
    """
    name = "hoi_reward"
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.iou_threshold = kwargs.get('iou_threshold', 0.5)
        
        # Lazy-load BERTScore model
        self._bertscore_model = None
    
    @property
    def bertscore_model(self):
        """Lazy-load BERTScore model on first use."""
        if self._bertscore_model is None:
            self._bertscore_model = get_bertscore_model()
        return self._bertscore_model
    
    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute rewards for batch of samples.
        
        For grounding tasks: Binary IoU >= 0.5
        For referring tasks: Hybrid BERTScore
        """
        # If rm_scores already exist, return them
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            score = {}
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode response
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Get ground truth and task type from reward_model
            reward_model = data_item.non_tensor_batch['reward_model']
            ground_truth = reward_model['ground_truth']
            task_type = reward_model.get('task_type', 'grounding')

            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, 'hoi_detection')

            # Compute score based on task type
            if task_type == 'grounding':
                # Ground truth is list of bounding boxes
                if isinstance(ground_truth, str):
                    try:
                        ground_truth = json.loads(ground_truth)
                    except:
                        ground_truth = []
                
                # Extract boxes from ground truth format
                gt_boxes = []
                if isinstance(ground_truth, list):
                    for item in ground_truth:
                        if isinstance(item, dict) and 'bbox_2d' in item:
                            gt_boxes.append(item['bbox_2d'])
                        elif isinstance(item, list) and len(item) == 4:
                            gt_boxes.append(item)
                
                task_score = grounding_score(response_str, gt_boxes, self.iou_threshold)
                score['grounding_score'] = task_score
                
            else:  # referring
                # Ground truth is action phrase string
                if isinstance(ground_truth, list):
                    ground_truth = ground_truth[0] if ground_truth else ""
                
                task_score = referring_score(response_str, str(ground_truth), self.bertscore_model)
                score['referring_score'] = task_score

            score['accuracy'] = 1.0 if task_score > 0 else 0.0
            score['score'] = task_score
            score['task_type'] = task_type

            # Track response lengths
            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length.item())
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length.item())

            reward = score["score"]
            
            # Store score info
            for key, value in score.items():
                if key != 'task_type':
                    reward_extra_info[key].append(value)
                else:
                    reward_extra_info[key].append(value)

            # For validation, use accuracy
            if self.num_examine == 1:
                reward = score["accuracy"]

            reward_tensor[i, valid_response_length - 1] = reward

            # Logging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[task_type] {task_type}")
                print(f"[prompt] {prompt_str[-500:]}")
                print(f"[response] {response_str[-500:]}")
                print(f"[ground_truth] {ground_truth}")
                for key, value in score.items():
                    print(f"[{key}] {value}")

        # Compute mean response lengths
        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        else:
            return reward_tensor


def clear_bertscore_cache():
    """Clear the cached BERTScore model to free memory."""
    global _bertscore_model, _bertscore_device
    
    if _bertscore_model is not None:
        del _bertscore_model
        _bertscore_model = None
        _bertscore_device = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("BERTScore model cache cleared")

