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
HOI Reward Manager V2 for Human-Object Interaction Detection.

IMPROVEMENTS OVER V1:
1. Verb-First Scoring: Action verb MUST match for any positive reward
2. Strict Exact Match Bonus: Full 1.0 only for exact phrase match
3. Better Action Phrase Extraction: Multiple strategies to find the action

Reward Formula for Referring Task:
- If verb (first word) doesn't match: 0.0 (no reward for wrong action)
- If verb matches + exact phrase match: 1.0
- If verb matches + partial object match: 0.5 + 0.5 * object_overlap

This addresses the key issue where "riding horse" vs "holding horse" 
previously got 0.5 reward (now gets 0.0).
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


# ============================================================================
# Common Action Verbs for Normalization
# ============================================================================

# Map common verb forms to base form
VERB_NORMALIZATION = {
    # -ing forms
    'riding': 'ride', 'sitting': 'sit', 'holding': 'hold', 'carrying': 'carry',
    'eating': 'eat', 'drinking': 'drink', 'reading': 'read', 'watching': 'watch',
    'using': 'use', 'playing': 'play', 'walking': 'walk', 'running': 'run',
    'standing': 'stand', 'lying': 'lie', 'pushing': 'push', 'pulling': 'pull',
    'throwing': 'throw', 'catching': 'catch', 'kicking': 'kick', 'hitting': 'hit',
    'cutting': 'cut', 'washing': 'wash', 'cleaning': 'clean', 'cooking': 'cook',
    'driving': 'drive', 'flying': 'fly', 'swimming': 'swim', 'climbing': 'climb',
    'jumping': 'jump', 'dancing': 'dance', 'singing': 'sing', 'writing': 'write',
    'drawing': 'draw', 'painting': 'paint', 'typing': 'type', 'talking': 'talk',
    'listening': 'listen', 'looking': 'look', 'waiting': 'wait', 'sleeping': 'sleep',
    'opening': 'open', 'closing': 'close', 'turning': 'turn', 'moving': 'move',
    'lifting': 'lift', 'dropping': 'drop', 'picking': 'pick', 'putting': 'put',
    'taking': 'take', 'giving': 'give', 'feeding': 'feed', 'petting': 'pet',
    'leading': 'lead', 'training': 'train', 'brushing': 'brush', 'grooming': 'groom',
    'wearing': 'wear', 'charging': 'charge', 'grilling': 'grill', 'skating': 'skate',
    # Past tense forms
    'rode': 'ride', 'sat': 'sit', 'held': 'hold', 'carried': 'carry',
    'ate': 'eat', 'drank': 'drink', 'read': 'read', 'watched': 'watch',
}

# Synonymous verbs (for fuzzy matching)
VERB_SYNONYMS = {
    'hold': ['carry', 'grip', 'grasp'],
    'carry': ['hold', 'transport', 'bring'],
    'ride': ['mount', 'sit on'],
    'sit': ['seat', 'rest'],
    'stand': ['rise', 'get up'],
    'watch': ['look at', 'observe', 'view'],
    'eat': ['consume', 'have', 'bite'],
    'use': ['operate', 'handle', 'work with'],
}


def normalize_verb(verb: str) -> str:
    """Normalize verb to base form."""
    verb = verb.lower().strip()
    return VERB_NORMALIZATION.get(verb, verb)


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


def extract_action_phrase(response_str: str) -> str:
    """
    Extract the action phrase from model response using multiple strategies.
    
    Strategies (in order of priority):
    1. "Final Answer:" pattern
    2. "ACTION:" pattern  
    3. "action phrase is:" pattern
    4. Last short line (2-4 words ending in -ing + noun)
    5. Verb + object pattern in last 100 chars
    
    Returns:
        Extracted action phrase, cleaned and lowercased
    """
    response_clean = clean_text(response_str)
    
    # Strategy 1: Final Answer pattern
    match = re.search(r'final\s*answer[:\.\s]+([^\n]+)', response_clean, re.IGNORECASE)
    if match:
        phrase = match.group(1).strip()
        # Remove any trailing punctuation or markdown
        phrase = re.sub(r'[\.!?,;:\*]+$', '', phrase)
        if 1 <= len(phrase.split()) <= 5:
            return phrase
    
    # Strategy 2: ACTION: pattern
    match = re.search(r'action[:\s]+([^\n]+)', response_clean, re.IGNORECASE)
    if match:
        phrase = match.group(1).strip()
        if 'phrase' not in phrase and 1 <= len(phrase.split()) <= 5:
            return phrase
    
    # Strategy 3: "action phrase is:" pattern
    match = re.search(r'action\s+phrase\s+is[:\s]+([^\n]+)', response_clean, re.IGNORECASE)
    if match:
        phrase = match.group(1).strip()
        phrase = re.sub(r'[\.!?,;:\*]+$', '', phrase)
        if 1 <= len(phrase.split()) <= 5:
            return phrase
    
    # Strategy 4: Last short line that looks like an action phrase
    lines = [l.strip() for l in response_clean.split('\n') if l.strip()]
    for line in reversed(lines):
        words = line.split()
        if 2 <= len(words) <= 4:
            # Check if first word is a verb (ends in -ing or is known verb)
            first_word = words[0]
            if first_word.endswith('ing') or first_word in VERB_NORMALIZATION:
                return line
    
    # Strategy 5: Find verb + noun pattern in last part of response
    last_chunk = response_clean[-200:] if len(response_clean) > 200 else response_clean
    verb_pattern = r'\b(sitting|holding|riding|carrying|eating|drinking|reading|watching|using|playing|walking|running|standing|pushing|pulling|throwing|catching|kicking|hitting|cutting|washing|cleaning|cooking|driving|flying|swimming|climbing|jumping|wearing|taking|giving|feeding)\s+(?:on\s+|with\s+|at\s+|in\s+|to\s+)?(\w+(?:\s+\w+)?)\b'
    
    matches = list(re.finditer(verb_pattern, last_chunk))
    if matches:
        last_match = matches[-1]
        return last_match.group(0)
    
    # Fallback: return the whole cleaned response (will likely fail matching)
    return response_clean


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


def referring_score_v2(response_str: str, ground_truth: str) -> Tuple[float, Dict[str, Any]]:
    """
    V2: Compute referring score with VERB-FIRST priority.
    
    Key improvement: The action verb (first word) MUST match for any positive reward.
    This prevents the model from getting reward for wrong actions like
    "riding horse" when ground truth is "holding horse".
    
    Scoring Logic:
    1. Extract action phrase from response
    2. Compare normalized verbs (first word)
    3. If verbs don't match -> 0.0 (strict!)
    4. If verbs match + full phrase match -> 1.0
    5. If verbs match + partial object match -> 0.5 + 0.5 * object_overlap
    
    Args:
        response_str: Model response containing action phrase
        ground_truth: Ground truth action phrase
        
    Returns:
        Tuple of (score, debug_info)
    """
    debug_info = {}
    
    # Extract and clean
    pred_phrase = extract_action_phrase(response_str)
    gt_clean = clean_text(ground_truth)
    
    debug_info['extracted_phrase'] = pred_phrase
    debug_info['ground_truth'] = gt_clean
    
    # Exact match -> 1.0
    if pred_phrase == gt_clean:
        debug_info['match_type'] = 'exact'
        return 1.0, debug_info
    
    # Split into words
    pred_words = pred_phrase.split()
    gt_words = gt_clean.split()
    
    if not pred_words or not gt_words:
        debug_info['match_type'] = 'empty'
        return 0.0, debug_info
    
    # Get verbs (first word) and normalize
    pred_verb = normalize_verb(pred_words[0])
    gt_verb = normalize_verb(gt_words[0])
    
    debug_info['pred_verb'] = pred_verb
    debug_info['gt_verb'] = gt_verb
    
    # CRITICAL: Verb MUST match for any positive reward
    if pred_verb != gt_verb:
        # Check if they're synonyms
        is_synonym = False
        if gt_verb in VERB_SYNONYMS:
            if pred_verb in VERB_SYNONYMS[gt_verb]:
                is_synonym = True
        if pred_verb in VERB_SYNONYMS:
            if gt_verb in VERB_SYNONYMS[pred_verb]:
                is_synonym = True
        
        if not is_synonym:
            debug_info['match_type'] = 'verb_mismatch'
            return 0.0, debug_info
        else:
            debug_info['match_type'] = 'verb_synonym'
    
    # Verb matches! Now check the rest
    pred_rest = set(pred_words[1:]) if len(pred_words) > 1 else set()
    gt_rest = set(gt_words[1:]) if len(gt_words) > 1 else set()
    
    # Full match after verb?
    if pred_rest == gt_rest:
        debug_info['match_type'] = 'full_match'
        return 1.0, debug_info
    
    # Partial object match
    if gt_rest:
        overlap = len(pred_rest & gt_rest) / len(gt_rest)
    else:
        overlap = 1.0  # No object to match, verb is enough
    
    # Score: 0.5 (verb correct) + 0.5 * object_overlap
    score = 0.5 + 0.5 * overlap
    debug_info['match_type'] = 'partial_match'
    debug_info['object_overlap'] = overlap
    
    return score, debug_info


@register("hoi_reward_v2")
class HOIRewardManagerV2:
    """
    HOI Reward Manager V2 with VERB-FIRST scoring for referring tasks.
    
    Key Improvements:
    - Referring: Verb must match for positive reward (fixes "riding" vs "holding" issue)
    - Grounding: Same binary IoU >= 0.5 reward (already working well)
    - Better action phrase extraction from verbose model outputs
    - Debug info logging for analysis
    """
    name = "hoi_reward_v2"
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.iou_threshold = kwargs.get('iou_threshold', 0.5)
        logger.info("HOI Reward Manager V2 initialized with verb-first scoring")
    
    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute rewards for batch of samples.
        
        For grounding tasks: Binary IoU >= 0.5
        For referring tasks: Verb-first scoring (V2)
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
                score['referring_score'] = 0.0  # Not applicable for grounding
                score['verb_match'] = 1.0  # N/A for grounding
                
            else:  # referring
                # Ground truth is action phrase string
                if isinstance(ground_truth, list):
                    ground_truth = ground_truth[0] if ground_truth else ""
                
                task_score, debug_info = referring_score_v2(response_str, str(ground_truth))
                score['referring_score'] = task_score
                score['grounding_score'] = 0.0  # Not applicable for referring
                
                # Track verb matching for analysis
                score['verb_match'] = 1.0 if debug_info.get('match_type') not in ['verb_mismatch', 'empty'] else 0.0
                score['match_type'] = debug_info.get('match_type', 'unknown')
                score['extracted_phrase'] = debug_info.get('extracted_phrase', '')[:50]

            score['accuracy'] = 1.0 if task_score > 0.5 else 0.0  # Changed threshold to 0.5
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


def clear_cache():
    """Clear any cached resources to free memory."""
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cache cleared")

