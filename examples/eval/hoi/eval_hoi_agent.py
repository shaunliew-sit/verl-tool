#!/usr/bin/env python3
"""
HOI Detection Agent Evaluation with Tool Calling (Multi-GPU + Thinking Logs)

Evaluates a trained HOI model with full tool-calling capability.
Uses vLLM server for inference and captures detailed thinking/reasoning logs.

Supports:
- Grounding tasks: Find person-object pairs for given action
- Referring tasks: Predict action from given person-object pair
- Multi-GPU parallel evaluation with async concurrency
- Thinking process logging for verification
- W&B integration for metrics tracking

Usage:
    # Start vLLM server with tensor parallelism (GPUs 0-3)
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/eval/hoi/start_vllm_server.sh \\
        checkpoints/.../actor/huggingface 8000 hoi-trained 4

    # Run grounding evaluation
    python examples/eval/hoi/eval_hoi_agent.py \\
        --task grounding \\
        --dataset hico \\
        --ann-file data/benchmarks_simplified/hico_ground_test_simplified.json \\
        --img-prefix data/hico_20160224_det/images/test2015 \\
        --endpoint http://localhost:8000/v1 \\
        --model hoi-trained \\
        --num-workers 4 --concurrency 8 \\
        --output-dir results/hico_ground \\
        --save-thinking --verbose --wandb

    # Run referring evaluation  
    python examples/eval/hoi/eval_hoi_agent.py \\
        --task referring \\
        --dataset hico \\
        --ann-file data/benchmarks_simplified/hico_action_referring_test_simplified.json \\
        --img-prefix data/hico_20160224_det/images/test2015 \\
        --endpoint http://localhost:8000/v1 \\
        --model hoi-trained \\
        --num-workers 4 --concurrency 8 \\
        --bertscore-gpu 4 \\
        --output-dir results/hico_referring \\
        --save-thinking --verbose --wandb
"""

import os
import sys
import json
import argparse
import asyncio
import aiohttp
import base64
import re
import logging
from io import BytesIO
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging - will be configured with file handler in main()
logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, log_filename: str = "eval.log") -> str:
    """
    Setup logging to both console and file.
    
    Args:
        output_dir: Directory to save the log file
        log_filename: Name of the log file
    
    Returns:
        Path to the log file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, log_filename)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Set root logger level
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler - captures everything
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    return log_path


# =============================================================================
# Utility Functions
# =============================================================================

def get_unique_output_dir(base_dir: str) -> str:
    """Generate unique output directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_dir}_{timestamp}"


# =============================================================================
# Constants
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "zoom_in",
            "description": "Zoom into a specific region of the image to examine details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Bounding box [x1, y1, x2, y2] in pixel coordinates"
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
            "description": "Zoom out to see the full image.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": "Detect objects in the current image view using Grounding DINO.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Object categories to detect, separated by ' . '"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentState:
    """State for agent conversation."""
    messages: List[Dict] = field(default_factory=list)
    current_image: Image.Image = None
    original_image: Image.Image = None
    tool_calls: List[Dict] = field(default_factory=list)
    zoom_history: List[List[int]] = field(default_factory=list)
    thinking_blocks: List[str] = field(default_factory=list)


@dataclass
class ThinkingLog:
    """Log entry for agent thinking process."""
    sample_id: int
    file_name: str
    task_type: str
    ground_truth: Dict
    conversation: List[Dict]
    thinking_blocks: List[str]
    tool_calls: List[Dict]
    num_turns: int
    num_tool_calls: int
    tools_used: List[str]
    prediction: str
    metrics: Dict


@dataclass
class GroundingSample:
    """Grounding task sample from simplified JSON."""
    file_name: str
    width: int
    height: int
    boxes: List[List[int]]
    action: str
    object_category: str
    num_pairs: int
    gt_box_inds: List[int]
    action_object_id: str = ""
    original_image_id: int = 0
    is_person_person: bool = False


@dataclass
class ReferringSample:
    """Referring task sample from simplified JSON."""
    file_name: str
    width: int
    height: int
    boxes: List[List[int]]
    person_box_idx: int
    object_box_idx: int
    gt_action: str


# =============================================================================
# Data Loading
# =============================================================================

def load_grounding_annotations(ann_file: str) -> List[GroundingSample]:
    """Load grounding task annotations from simplified JSON."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        samples.append(GroundingSample(
            file_name=item['file_name'],
            width=item['width'],
            height=item['height'],
            boxes=item['boxes'],
            action=item['action'],
            object_category=item['object_category'],
            num_pairs=item['num_pairs'],
            gt_box_inds=item['gt_box_inds'],
            action_object_id=item.get('action_object_id', ''),
            original_image_id=item.get('original_image_id', 0),
            is_person_person=item.get('is_person_person', False)
        ))
    return samples


def load_referring_annotations(ann_file: str) -> List[ReferringSample]:
    """Load referring task annotations from simplified JSON."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        samples.append(ReferringSample(
            file_name=item['file_name'],
            width=item['width'],
            height=item['height'],
            boxes=item['boxes'],
            person_box_idx=item['person_box_idx'],
            object_box_idx=item['object_box_idx'],
            gt_action=item['gt_action']
        ))
    return samples


# =============================================================================
# Prompt Building
# =============================================================================

def build_grounding_prompt(action: str, object_category: str) -> str:
    """Build grounding task prompt."""
    return f"""Human-Object Interaction Detection Task: Find all instances of "{action} {object_category}" in this image.

For each interaction found, output the bounding boxes for:
1. The PERSON performing the action
2. The {object_category.upper()} involved in the action

Output format: List of {{"bbox_2d": [x1, y1, x2, y2], "label": "person/object"}} pairs.

Guidelines: Analyze the image to locate human-object interaction pairs. You may use zoom_in to examine details or detect_objects to find candidates. For each person-object pair performing "{action}", output their bounding boxes in JSON format. Coordinates should be in the 1000x1000 normalized format."""


def build_referring_prompt(person_box: List[int], object_box: List[int], 
                          object_category: str, width: int, height: int) -> str:
    """Build referring task prompt with normalized coordinates."""
    # Normalize to 1000-scale
    def normalize(box, w, h):
        return [
            int(box[0] * 1000 / w),
            int(box[1] * 1000 / h),
            int(box[2] * 1000 / w),
            int(box[3] * 1000 / h)
        ]
    
    person_norm = normalize(person_box, width, height)
    object_norm = normalize(object_box, width, height)
    
    return f"""Action Recognition Task: The first region {{"bbox_2d": {person_norm}, "label": "person"}} contains a PERSON. The second region {{"bbox_2d": {object_norm}, "label": "{object_category}"}} contains an OBJECT. Describe the action the person is performing with this object. Respond with only the action phrase (e.g., "riding bicycle", "sitting on bench").

Guidelines: Analyze the provided bounding boxes to determine what action the person is performing with the object. You may use zoom_in to examine interaction details. Output only the action phrase (e.g., "riding bicycle", "sitting on bench"). Use base verb form without articles."""


# =============================================================================
# Agent Evaluator
# =============================================================================

class HOIAgentEvaluator:
    """Evaluator that runs the model with tool calling."""
    
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        max_turns: int = 10,
        max_tokens: int = 2048,
        verbose: bool = False,
        save_thinking: bool = True
    ):
        self.endpoint = endpoint.rstrip('/')
        self.model_name = model_name
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.save_thinking = save_thinking
        self.hoi_tool = None
        
        try:
            from verl_tool.servers.tools.hoi_detector import HOIDetectorTool
            self.hoi_tool = HOIDetectorTool()
        except Exception:
            pass  # Tool not needed if just testing model responses
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string."""
        buffered = BytesIO()
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    async def call_model(self, messages: List[Dict], images: List[Image.Image] = None,
                         session: aiohttp.ClientSession = None) -> Dict:
        """Call the vLLM server with messages and optional images."""
        formatted_messages = []
        images_added = False
        
        for msg in messages:
            if msg['role'] == 'system':
                formatted_messages.append({'role': 'system', 'content': msg['content']})
            elif msg['role'] == 'user':
                content = msg['content']
                if images and not images_added and not isinstance(content, list):
                    content_parts = []
                    for img in images:
                        img_base64 = self._encode_image(img)
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        })
                    content_parts.append({"type": "text", "text": content})
                    content = content_parts
                    images_added = True
                formatted_messages.append({'role': 'user', 'content': content})
            elif msg['role'] == 'assistant':
                formatted_messages.append({
                    'role': 'assistant',
                    'content': msg.get('content', ''),
                    'tool_calls': msg.get('tool_calls')
                })
            elif msg['role'] == 'tool':
                formatted_messages.append({
                    'role': 'tool',
                    'tool_call_id': msg.get('tool_call_id', ''),
                    'content': msg['content']
                })
        
        payload = {
            'model': self.model_name,
            'messages': formatted_messages,
            'tools': TOOL_DEFINITIONS,
            'tool_choice': 'auto',
            'max_tokens': self.max_tokens,
            'temperature': 0.7,
            'repetition_penalty': 1.1,
        }
        
        should_close = False
        if session is None:
            session = aiohttp.ClientSession()
            should_close = True
        
        try:
            async with session.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                return await response.json()
        finally:
            if should_close:
                await session.close()
    
    def execute_tool(self, tool_name: str, arguments: Dict, state: AgentState) -> str:
        """Execute a tool and return the result."""
        if tool_name == 'zoom_in':
            bbox = arguments.get('bbox', arguments.get('bbox_2d', []))
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(c) for c in bbox]
                # Ensure valid crop region
                w, h = state.current_image.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    cropped = state.current_image.crop((x1, y1, x2, y2))
                    state.current_image = cropped
                    state.zoom_history.append(bbox)
                    return f"Zoomed into region {bbox}. Now viewing a {cropped.size[0]}x{cropped.size[1]} cropped area."
            return "Invalid bbox format. Expected [x1, y1, x2, y2]."
        
        elif tool_name == 'zoom_out':
            state.current_image = state.original_image.copy()
            state.zoom_history.clear()
            return f"Zoomed out to full image view ({state.original_image.size[0]}x{state.original_image.size[1]})."
        
        elif tool_name == 'detect_objects':
            query = arguments.get('query', arguments.get('class_names', 'person . object'))
            try:
                from verl_tool.servers.tools.hoi_detector import detect_objects_with_grounding_dino
                detections = detect_objects_with_grounding_dino(state.current_image, query)
                if detections:
                    result = "Detected objects:\n"
                    for det in detections:
                        result += f"  - {det['label']}: bbox={det['bbox']}, confidence={det['confidence']:.2f}\n"
                    return result
                return "No objects detected matching the query."
            except Exception as e:
                return f"Detection failed: {str(e)}"
        
        return f"Unknown tool: {tool_name}"
    
    def _extract_thinking_blocks(self, content: str) -> List[str]:
        """Extract <think>...</think> blocks from content."""
        blocks = []
        pattern = r'<think>(.*?)</think>'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            blocks.append(match.strip())
        return blocks
    
    def _extract_tool_calls_from_content(self, content: str) -> List[Dict]:
        """Extract XML-style tool calls from content."""
        tool_calls = []
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                tc_data = json.loads(match)
                tool_calls.append({
                    'id': f'xml_{len(tool_calls)}',
                    'function': {
                        'name': tc_data.get('name', ''),
                        'arguments': json.dumps(tc_data.get('arguments', {}))
                    }
                })
            except json.JSONDecodeError:
                pass
        return tool_calls
    
    async def run_agent_loop(
        self,
        image_path: str,
        prompt: str,
        session: aiohttp.ClientSession = None
    ) -> Dict:
        """Run the agent loop for a single sample."""
        image = Image.open(image_path).convert('RGB')
        
        state = AgentState(
            current_image=image.copy(),
            original_image=image.copy(),
        )
        
        state.messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ]
        
        final_response = ""
        turn = 0
        
        for turn in range(self.max_turns):
            try:
                response = await self.call_model(state.messages, [state.current_image], session)
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error calling model: {e}")
                break
            
            choice = response.get('choices', [{}])[0]
            message = choice.get('message', {})
            content = message.get('content', '') or ''
            
            # Extract thinking blocks
            thinking = self._extract_thinking_blocks(content)
            state.thinking_blocks.extend(thinking)
            
            # Check for tool calls
            tool_calls = message.get('tool_calls', [])
            if not tool_calls and content:
                tool_calls = self._extract_tool_calls_from_content(content)
            
            # Check for repetition
            if content and len(content) > 500:
                repeat_pattern = r'(.{20,}?)\1{5,}'
                if re.search(repeat_pattern, content):
                    lines = content.split('\n')
                    unique = []
                    seen = set()
                    for line in lines:
                        ls = line.strip()
                        if ls and ls not in seen:
                            unique.append(line)
                            seen.add(ls)
                    final_response = '\n'.join(unique[:10])
                    break
            
            if tool_calls:
                state.messages.append({
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': tool_calls
                })
                
                for tc in tool_calls:
                    func = tc.get('function', {})
                    tool_name = func.get('name', '')
                    try:
                        arguments = json.loads(func.get('arguments', '{}'))
                    except:
                        arguments = {}
                    
                    result = self.execute_tool(tool_name, arguments, state)
                    state.tool_calls.append({
                        'turn': turn + 1,
                        'name': tool_name,
                        'args': arguments,
                        'result': result
                    })
                    
                    state.messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.get('id', ''),
                        'name': tool_name,
                        'content': result
                    })
            else:
                final_response = content
                state.messages.append({'role': 'assistant', 'content': content})
                break
        
        tools_used = list(set(tc['name'] for tc in state.tool_calls))
        
        return {
            'response': final_response,
            'tool_calls': state.tool_calls,
            'num_turns': turn + 1,
            'zoom_history': state.zoom_history,
            'thinking_blocks': state.thinking_blocks,
            'tools_used': tools_used,
            'conversation': state.messages if self.save_thinking else []
        }


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def get_box_area(box: List[int]) -> float:
    """Get box area."""
    return (box[2] - box[0]) * (box[3] - box[1])


def convert_normalized_to_pixel(boxes: List[Dict], img_width: int, img_height: int, 
                                  normalized_size: int = 1000) -> List[Dict]:
    """Convert boxes from normalized coordinates (e.g., 1000x1000) to pixel coordinates.
    
    Detects if boxes appear to be in normalized format based on their coordinate values.
    """
    if not boxes:
        return boxes
    
    # Check if boxes appear to be in normalized format
    # If any coordinate is > max(img_width, img_height) * 1.5, likely normalized
    max_coord = max(
        max(b['bbox_2d']) for b in boxes if b.get('bbox_2d') and len(b['bbox_2d']) == 4
    ) if boxes else 0
    
    img_max = max(img_width, img_height)
    
    # If max coordinate is much larger than image size, assume normalized format
    if max_coord > img_max * 1.5 and max_coord <= normalized_size * 1.1:
        converted = []
        for box in boxes:
            if 'bbox_2d' in box and len(box['bbox_2d']) == 4:
                x1, y1, x2, y2 = box['bbox_2d']
                converted.append({
                    "bbox_2d": [
                        int(x1 * img_width / normalized_size),
                        int(y1 * img_height / normalized_size),
                        int(x2 * img_width / normalized_size),
                        int(y2 * img_height / normalized_size)
                    ],
                    "label": box.get('label', 'object')
                })
            else:
                converted.append(box)
        return converted
    
    return boxes


def extract_boxes_from_response(response: str) -> List[Dict]:
    """Extract bounding boxes from model response."""
    boxes = []
    
    # Pattern for {"bbox_2d": [...], "label": "..."}
    pattern = r'\{\s*"bbox_2d"\s*:\s*\[([^\]]+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, response)
    
    for coords_str, label in matches:
        try:
            coords = [int(float(x.strip())) for x in coords_str.split(',')]
            if len(coords) == 4:
                boxes.append({"bbox_2d": coords, "label": label.lower()})
        except:
            continue
    
    # Fallback: try to find [x1, y1, x2, y2] patterns
    if not boxes:
        bbox_pattern = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
        bbox_matches = re.findall(bbox_pattern, response)
        for i, (x1, y1, x2, y2) in enumerate(bbox_matches):
            label = "person" if i % 2 == 0 else "object"
            boxes.append({"bbox_2d": [int(x1), int(y1), int(x2), int(y2)], "label": label})
    
    return boxes


def extract_pairs_from_boxes(boxes: List[Dict]) -> List[Tuple[List[int], List[int]]]:
    """Extract person-object pairs from boxes."""
    persons = [b['bbox_2d'] for b in boxes if 'person' in b['label'].lower()]
    objects = [b['bbox_2d'] for b in boxes if 'person' not in b['label'].lower()]
    
    pairs = []
    for p in persons:
        for o in objects:
            pairs.append((p, o))
    
    # If no clear distinction, pair sequentially
    if not pairs and len(boxes) >= 2:
        for i in range(0, len(boxes) - 1, 2):
            pairs.append((boxes[i]['bbox_2d'], boxes[i+1]['bbox_2d']))
    
    return pairs


def match_pairs_greedy(pred_pairs: List[Tuple], gt_pairs: List[Tuple], 
                       iou_threshold: float = 0.5) -> int:
    """Greedy matching of predicted pairs to ground truth pairs."""
    if not pred_pairs or not gt_pairs:
        return 0
    
    matched = 0
    used_gt = set()
    
    for pred_p, pred_o in pred_pairs:
        best_gt_idx = -1
        best_score = 0
        
        for gt_idx, (gt_p, gt_o) in enumerate(gt_pairs):
            if gt_idx in used_gt:
                continue
            
            iou_p = compute_iou(pred_p, gt_p)
            iou_o = compute_iou(pred_o, gt_o)
            
            if iou_p >= iou_threshold and iou_o >= iou_threshold:
                score = iou_p + iou_o
                if score > best_score:
                    best_score = score
                    best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matched += 1
            used_gt.add(best_gt_idx)
    
    return matched


def compute_grounding_metrics(predictions: List[Dict], iou_thresholds: List[float] = None) -> Dict:
    """Compute grounding metrics (AR at multiple IoU thresholds)."""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    recalls_per_threshold = {t: [] for t in iou_thresholds}
    recalls_by_size = {'small': [], 'medium': [], 'large': []}
    
    for pred in predictions:
        gt_pairs = pred.get('gt_pairs', [])
        pred_pairs = pred.get('pred_pairs', [])
        
        if not gt_pairs:
            continue
        
        # Compute recall at each IoU threshold
        for threshold in iou_thresholds:
            matched = match_pairs_greedy(pred_pairs, gt_pairs, threshold)
            recall = matched / len(gt_pairs)
            recalls_per_threshold[threshold].append(recall)
        
        # Size-based metrics (using person box area)
        for gt_p, gt_o in gt_pairs:
            area = get_box_area(gt_p)
            if area < 32 * 32:
                size = 'small'
            elif area < 96 * 96:
                size = 'medium'
            else:
                size = 'large'
            
            # Check if this pair was matched at IoU 0.5
            matched_this = False
            for pred_p, pred_o in pred_pairs:
                if compute_iou(pred_p, gt_p) >= 0.5 and compute_iou(pred_o, gt_o) >= 0.5:
                    matched_this = True
                    break
            recalls_by_size[size].append(1.0 if matched_this else 0.0)
    
    metrics = {}
    
    # Average across thresholds
    all_recalls = []
    for threshold in iou_thresholds:
        if recalls_per_threshold[threshold]:
            ar = np.mean(recalls_per_threshold[threshold])
            metrics[f'AR@{threshold}'] = ar
            all_recalls.append(ar)
    
    if all_recalls:
        metrics['AR'] = np.mean(all_recalls)
    
    # Size-based
    for size in ['small', 'medium', 'large']:
        if recalls_by_size[size]:
            metrics[f'AR{size[0]}'] = np.mean(recalls_by_size[size])
    
    return metrics


def clean_action_text(text: str) -> str:
    """Clean action text for comparison."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove markdown bold
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)  # Remove think blocks
    text = ' '.join(text.split())
    return text.strip()


def extract_action_from_response(response: str) -> str:
    """Extract action phrase from model response."""
    # First, detect and truncate repetitive patterns (model degeneration)
    # Look for patterns like "upgrading upgrading upgrading" or similar
    words = response.split()
    for i in range(min(50, len(words))):
        word = words[i] if i < len(words) else ""
        # Check if this word repeats many times
        if word and words[i:i+10].count(word) >= 5:
            # Truncate before the repetition
            response = ' '.join(words[:i])
            break
    
    response = clean_action_text(response)
    
    # Strategy 1: Look for "Final Answer:" pattern
    match = re.search(r'final\s*answer[:\.\s]+\s*([^\n]+)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip('.!?,;:')
    
    # Strategy 2: Look for "action:" pattern
    match = re.search(r'action[:\.\s]+\s*([^\n]+)', response, re.IGNORECASE)
    if match:
        result = match.group(1).strip()
        if result and result not in ['phrase is', 'is', 'phrase']:
            return result.rstrip('.!?,;:')
    
    # Strategy 3: Look for common action phrases at the beginning
    # Format: "verb object" or "verb preposition object"
    common_verbs = ['riding', 'sitting', 'holding', 'walking', 'carrying', 'wearing', 
                    'playing', 'eating', 'drinking', 'reading', 'using', 'operating',
                    'driving', 'lying', 'standing', 'throwing', 'catching', 'pushing',
                    'pulling', 'cutting', 'washing', 'cleaning', 'repairing', 'fixing']
    for verb in common_verbs:
        pattern = rf'\b({verb}\s+(?:on\s+|with\s+|in\s+)?[a-z]+(?:\s+[a-z]+)?)\b'
        match = re.search(pattern, response.lower())
        if match:
            return match.group(1).strip()
    
    # Strategy 4: Look for short action-like lines at the beginning
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    for line in lines[:3]:  # Check first 3 lines
        if len(line) < 50 and not line.startswith('#'):
            words = line.split()
            if 1 <= len(words) <= 5:
                return line.rstrip('.!?,;:')
    
    # Fallback: return cleaned response truncated
    return response[:100].rstrip('.!?,;:') if response else ""


def compute_referring_metrics(predictions: List[Dict], bertscore_gpu: int = 0) -> Dict:
    """Compute referring metrics (CIDEr, METEOR, BERTScore)."""
    pred_texts = []
    gt_texts = []
    
    for p in predictions:
        pred_texts.append(p.get('prediction', ''))
        gt_texts.append(p.get('ground_truth', ''))
    
    metrics = {}
    
    # Exact match
    exact_matches = sum(1 for p, g in zip(pred_texts, gt_texts) 
                       if clean_action_text(p) == clean_action_text(g))
    metrics['exact_match'] = exact_matches / len(predictions) if predictions else 0
    
    # Try METEOR
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        meteor_scores = []
        for pred, gt in zip(pred_texts, gt_texts):
            pred_clean = clean_action_text(pred)
            gt_clean = clean_action_text(gt)
            if pred_clean and gt_clean:
                try:
                    score = meteor_score([gt_clean.split()], pred_clean.split())
                    meteor_scores.append(score)
                except:
                    meteor_scores.append(0.0)
            else:
                meteor_scores.append(0.0)
        
        metrics['meteor'] = np.mean(meteor_scores) if meteor_scores else 0.0
    except ImportError:
        logger.warning("NLTK not available for METEOR score")
    
    # Try CIDEr
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # Format for CIDEr
        gts = {i: [clean_action_text(gt)] for i, gt in enumerate(gt_texts)}
        res = {i: [clean_action_text(pred)] for i, pred in enumerate(pred_texts)}
        
        cider = Cider()
        score, _ = cider.compute_score(gts, res)
        metrics['cider'] = score
    except ImportError:
        logger.warning("pycocoevalcap not available for CIDEr score")
    
    # Try BERTScore
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(bertscore_gpu)
        from bert_score import score as bert_score
        
        pred_clean = [clean_action_text(p) for p in pred_texts]
        gt_clean = [clean_action_text(g) for g in gt_texts]
        
        # Filter out empty strings
        valid_pairs = [(p, g) for p, g in zip(pred_clean, gt_clean) if p and g]
        
        if valid_pairs:
            preds, refs = zip(*valid_pairs)
            P, R, F1 = bert_score(
                list(preds), list(refs),
                model_type="roberta-large",
                lang="en",
                batch_size=64,
                rescale_with_baseline=True,
                verbose=False
            )
            
            metrics['bertscore_precision'] = float(P.mean())
            metrics['bertscore_recall'] = float(R.mean())
            metrics['bertscore_f1'] = float(F1.mean())
    except ImportError:
        logger.warning("bert_score not available")
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
    
    return metrics


# =============================================================================
# Visualization
# =============================================================================

def visualize_grounding(image_path: str, pred_boxes: List[Dict], gt_boxes: List[Dict],
                        save_path: str = None) -> Image.Image:
    """Create visualization with predicted and ground truth boxes."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Draw ground truth in green
    for box in gt_boxes:
        bbox = box.get('bbox_2d', box.get('bbox', []))
        label = box.get('label', 'gt')
        if len(bbox) == 4:
            draw.rectangle(bbox, outline=(0, 200, 0), width=3)
            draw.text((bbox[0] + 2, bbox[1] + 2), f"GT:{label}", fill=(0, 200, 0), font=font)
    
    # Draw predictions in red/blue
    colors = {'person': (255, 100, 100), 'object': (100, 100, 255)}
    for box in pred_boxes:
        bbox = box.get('bbox_2d', [])
        label = box.get('label', 'pred')
        if len(bbox) == 4:
            color = colors.get(label.lower(), (255, 165, 0))
            draw.rectangle(bbox, outline=color, width=2)
            draw.text((bbox[0] + 2, bbox[3] - 16), f"P:{label}", fill=color, font=font)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        img.save(save_path)
    
    return img


def visualize_referring(
    image_path: str,
    person_box: List[int],
    object_box: List[int],
    object_label: str,
    gt_action: str,
    pred_action: str,
    is_correct: bool,
    save_path: str = None
) -> Image.Image:
    """Create visualization for referring task with action comparison."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Try to load a larger font for text overlay
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        font_large = font
    
    # Draw person box in cyan
    if len(person_box) == 4:
        draw.rectangle(person_box, outline=(0, 255, 255), width=3)
        draw.text((person_box[0] + 2, person_box[1] + 2), "PERSON", fill=(0, 255, 255), font=font)
    
    # Draw object box in magenta
    if len(object_box) == 4:
        draw.rectangle(object_box, outline=(255, 0, 255), width=3)
        draw.text((object_box[0] + 2, object_box[1] + 2), object_label.upper(), fill=(255, 0, 255), font=font)
    
    # Add text overlay at the top
    img_width = img.size[0]
    
    # Draw semi-transparent background for text
    overlay_height = 80
    overlay = Image.new('RGBA', (img_width, overlay_height), (0, 0, 0, 180))
    img = img.convert('RGBA')
    img.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img)
    
    # Draw match indicator
    indicator = "OK" if is_correct else "X"
    indicator_color = (0, 255, 0) if is_correct else (255, 0, 0)
    draw.text((10, 10), indicator, fill=indicator_color, font=font_large)
    
    # Draw GT and Pred actions
    draw.text((50, 10), f"GT: {gt_action}", fill=(255, 255, 255), font=font)
    draw.text((50, 40), f"Pred: {pred_action}", fill=(255, 255, 100), font=font)
    
    # Convert back to RGB for saving
    img = img.convert('RGB')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        img.save(save_path)
    
    return img


# =============================================================================
# Batch Evaluation
# =============================================================================

async def evaluate_grounding_batch(
    evaluator: HOIAgentEvaluator,
    samples: List[GroundingSample],
    img_prefix: str,
    concurrency: int = 8,
    save_thinking: bool = True,
    save_visualizations: bool = False,
    output_dir: str = None,
    verbose: bool = False
) -> Tuple[List[Dict], List[ThinkingLog]]:
    """Evaluate grounding task on a batch of samples."""
    results = []
    thinking_logs = []
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create visualization directory if needed
    viz_dir = None
    if save_visualizations and output_dir:
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    
    async def process_one(idx: int, sample: GroundingSample, session: aiohttp.ClientSession):
        async with semaphore:
            image_path = os.path.join(img_prefix, sample.file_name)
            
            if not os.path.exists(image_path):
                return None, None
            
            # Build ground truth pairs
            gt_pairs = []
            for i in range(sample.num_pairs):
                person_idx = sample.gt_box_inds[i * 2]
                object_idx = sample.gt_box_inds[i * 2 + 1]
                gt_pairs.append((sample.boxes[person_idx], sample.boxes[object_idx]))
            
            prompt = build_grounding_prompt(sample.action, sample.object_category)
            
            try:
                result = await evaluator.run_agent_loop(image_path, prompt, session)
            except Exception as e:
                if verbose:
                    logger.error(f"Error processing {sample.file_name}: {e}")
                return None, None
            
            # Extract predictions and convert coordinates
            pred_boxes = extract_boxes_from_response(result['response'])
            
            # Get image dimensions for coordinate conversion
            try:
                img = Image.open(image_path)
                img_width, img_height = img.size
                img.close()
                
                # Always convert from 1000x1000 normalized to pixel coordinates
                # since training uses 1000x1000 format
                converted_boxes = []
                for box in pred_boxes:
                    if 'bbox_2d' in box and len(box['bbox_2d']) == 4:
                        x1, y1, x2, y2 = box['bbox_2d']
                        converted_boxes.append({
                            "bbox_2d": [
                                int(x1 * img_width / 1000),
                                int(y1 * img_height / 1000),
                                int(x2 * img_width / 1000),
                                int(y2 * img_height / 1000)
                            ],
                            "label": box.get('label', 'object')
                        })
                    else:
                        converted_boxes.append(box)
                pred_boxes = converted_boxes
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not load image dimensions for {image_path}: {e}")
            
            pred_pairs = extract_pairs_from_boxes(pred_boxes)
            
            # Compute per-sample metrics
            sample_metrics = {}
            for threshold in [0.5, 0.75]:
                matched = match_pairs_greedy(pred_pairs, gt_pairs, threshold)
                recall = matched / len(gt_pairs) if gt_pairs else 0
                sample_metrics[f'recall@{threshold}'] = recall
            
            result_entry = {
                'sample_id': idx,
                'file_name': sample.file_name,
                'action': sample.action,
                'object_category': sample.object_category,
                'gt_pairs': gt_pairs,
                'pred_pairs': pred_pairs,
                'pred_boxes': pred_boxes,
                'num_gt_pairs': len(gt_pairs),
                'num_pred_pairs': len(pred_pairs),
                'response': result['response'],
                'num_turns': result['num_turns'],
                'num_tool_calls': len(result['tool_calls']),
                'tools_used': result['tools_used'],
                'metrics': sample_metrics
            }
            
            # Create thinking log
            thinking_log = None
            if save_thinking:
                thinking_log = ThinkingLog(
                    sample_id=idx,
                    file_name=sample.file_name,
                    task_type='grounding',
                    ground_truth={
                        'action': sample.action,
                        'object': sample.object_category,
                        'pairs': gt_pairs
                    },
                    conversation=result.get('conversation', []),
                    thinking_blocks=result.get('thinking_blocks', []),
                    tool_calls=result['tool_calls'],
                    num_turns=result['num_turns'],
                    num_tool_calls=len(result['tool_calls']),
                    tools_used=result['tools_used'],
                    prediction=result['response'],
                    metrics=sample_metrics
                )
            
            # Save visualization if enabled
            if viz_dir:
                # Build GT boxes list for visualization
                gt_boxes = []
                for person_box, obj_box in gt_pairs:
                    gt_boxes.append({'bbox_2d': person_box, 'label': 'person'})
                    gt_boxes.append({'bbox_2d': obj_box, 'label': sample.object_category})
                
                # Generate visualization filename
                base_name = Path(sample.file_name).stem
                viz_path = os.path.join(viz_dir, f"{base_name}_{idx}.jpg")
                
                try:
                    visualize_grounding(image_path, pred_boxes, gt_boxes, save_path=viz_path)
                except Exception as e:
                    if verbose:
                        logger.warning(f"Failed to save visualization for {sample.file_name}: {e}")
            
            return result_entry, thinking_log
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_one(i, s, session) for i, s in enumerate(samples)]
        
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Grounding"):
            result, log = await coro
            if result is not None:
                results.append(result)
            if log is not None:
                thinking_logs.append(log)
    
    return results, thinking_logs


async def evaluate_referring_batch(
    evaluator: HOIAgentEvaluator,
    samples: List[ReferringSample],
    img_prefix: str,
    concurrency: int = 8,
    save_thinking: bool = True,
    save_visualizations: bool = False,
    output_dir: str = None,
    verbose: bool = False
) -> Tuple[List[Dict], List[ThinkingLog]]:
    """Evaluate referring task on a batch of samples."""
    results = []
    thinking_logs = []
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create visualization directory if needed
    viz_dir = None
    if save_visualizations and output_dir:
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    
    async def process_one(idx: int, sample: ReferringSample, session: aiohttp.ClientSession):
        async with semaphore:
            image_path = os.path.join(img_prefix, sample.file_name)
            
            if not os.path.exists(image_path):
                return None, None
            
            person_box = sample.boxes[sample.person_box_idx]
            object_box = sample.boxes[sample.object_box_idx]
            
            # Extract object category from gt_action (last word typically)
            words = sample.gt_action.split()
            object_category = words[-1] if words else "object"
            
            prompt = build_referring_prompt(
                person_box, object_box, object_category,
                sample.width, sample.height
            )
            
            try:
                result = await evaluator.run_agent_loop(image_path, prompt, session)
            except Exception as e:
                if verbose:
                    logger.error(f"Error processing {sample.file_name}: {e}")
                return None, None
            
            # Extract prediction
            prediction = extract_action_from_response(result['response'])
            gt = clean_action_text(sample.gt_action)
            pred_clean = clean_action_text(prediction)
            
            exact_match = pred_clean == gt
            
            result_entry = {
                'sample_id': idx,
                'file_name': sample.file_name,
                'ground_truth': sample.gt_action,
                'prediction': prediction,
                'exact_match': exact_match,
                'response': result['response'],
                'num_turns': result['num_turns'],
                'num_tool_calls': len(result['tool_calls']),
                'tools_used': result['tools_used']
            }
            
            thinking_log = None
            if save_thinking:
                thinking_log = ThinkingLog(
                    sample_id=idx,
                    file_name=sample.file_name,
                    task_type='referring',
                    ground_truth={'action': sample.gt_action},
                    conversation=result.get('conversation', []),
                    thinking_blocks=result.get('thinking_blocks', []),
                    tool_calls=result['tool_calls'],
                    num_turns=result['num_turns'],
                    num_tool_calls=len(result['tool_calls']),
                    tools_used=result['tools_used'],
                    prediction=prediction,
                    metrics={'exact_match': exact_match}
                )
            
            # Save visualization if enabled
            if viz_dir:
                base_name = Path(sample.file_name).stem
                viz_path = os.path.join(viz_dir, f"{base_name}_{idx}.jpg")
                
                try:
                    visualize_referring(
                        image_path=image_path,
                        person_box=person_box,
                        object_box=object_box,
                        object_label=object_category,
                        gt_action=sample.gt_action,
                        pred_action=prediction,
                        is_correct=exact_match,
                        save_path=viz_path
                    )
                except Exception as e:
                    if verbose:
                        logger.warning(f"Failed to save visualization for {sample.file_name}: {e}")
            
            return result_entry, thinking_log
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_one(i, s, session) for i, s in enumerate(samples)]
        
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Referring"):
            result, log = await coro
            if result is not None:
                results.append(result)
            if log is not None:
                thinking_logs.append(log)
    
    return results, thinking_logs


def compute_tool_usage_stats(thinking_logs: List[ThinkingLog]) -> Dict:
    """Compute aggregate tool usage statistics."""
    stats = {
        'total_samples': len(thinking_logs),
        'samples_with_tool_calls': 0,
        'tool_call_distribution': defaultdict(lambda: {'count': 0}),
        'turns_distribution': defaultdict(int),
        'thinking_block_stats': {
            'total_blocks': 0,
            'samples_with_thinking': 0,
            'avg_length': 0
        }
    }
    
    all_thinking_lengths = []
    
    for log in thinking_logs:
        if log.num_tool_calls > 0:
            stats['samples_with_tool_calls'] += 1
        
        for tc in log.tool_calls:
            stats['tool_call_distribution'][tc['name']]['count'] += 1
        
        turns_key = str(log.num_turns) if log.num_turns < 5 else '5+'
        stats['turns_distribution'][turns_key] += 1
        
        if log.thinking_blocks:
            stats['thinking_block_stats']['samples_with_thinking'] += 1
            stats['thinking_block_stats']['total_blocks'] += len(log.thinking_blocks)
            for block in log.thinking_blocks:
                all_thinking_lengths.append(len(block))
    
    # Compute averages
    for tool in stats['tool_call_distribution']:
        stats['tool_call_distribution'][tool]['avg_per_sample'] = \
            stats['tool_call_distribution'][tool]['count'] / len(thinking_logs) if thinking_logs else 0
    
    if all_thinking_lengths:
        stats['thinking_block_stats']['avg_length'] = np.mean(all_thinking_lengths)
    
    # Convert defaultdicts to regular dicts
    stats['tool_call_distribution'] = dict(stats['tool_call_distribution'])
    stats['turns_distribution'] = dict(stats['turns_distribution'])
    
    return stats


# =============================================================================
# Main Evaluation Functions
# =============================================================================

async def run_grounding_evaluation(args):
    """Run grounding evaluation."""
    logger.info("=" * 60)
    logger.info("HOI Grounding Evaluation")
    logger.info("=" * 60)
    logger.info(f"Annotation file: {args.ann_file}")
    logger.info(f"Image prefix: {args.img_prefix}")
    logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info("=" * 60)
    
    # Load annotations
    samples = load_grounding_annotations(args.ann_file)
    if args.max_images:
        samples = samples[:args.max_images]
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Create evaluator
    evaluator = HOIAgentEvaluator(
        endpoint=args.endpoint,
        model_name=args.model,
        max_turns=args.max_turns,
        verbose=args.verbose,
        save_thinking=args.save_thinking
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    results, thinking_logs = await evaluate_grounding_batch(
        evaluator, samples, args.img_prefix,
        concurrency=args.concurrency,
        save_thinking=args.save_thinking,
        save_visualizations=args.save_viz,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Compute metrics
    metrics = compute_grounding_metrics(results)
    
    # Compute tool usage stats
    tool_stats = compute_tool_usage_stats(thinking_logs)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    for key, value in sorted(metrics.items()):
        logger.info(f"{key}: {value:.4f}")
    logger.info("=" * 60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'per_sample_results.json'), 'w') as f:
        # Convert tuples to lists for JSON serialization
        serializable_results = []
        for r in results:
            r_copy = r.copy()
            r_copy['gt_pairs'] = [[list(p), list(o)] for p, o in r['gt_pairs']]
            r_copy['pred_pairs'] = [[list(p), list(o)] for p, o in r['pred_pairs']]
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'tool_usage_stats.json'), 'w') as f:
        json.dump(tool_stats, f, indent=2)
    
    if args.save_thinking and thinking_logs:
        with open(os.path.join(args.output_dir, 'thinking.jsonl'), 'w') as f:
            for log in thinking_logs:
                f.write(json.dumps(asdict(log)) + '\n')
    
    # W&B logging
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"grounding_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
            wandb.log(metrics)
            wandb.log({"tool_stats": tool_stats})
            wandb.finish()
        except ImportError:
            logger.warning("wandb not available")
    
    logger.info(f"\nResults saved to {args.output_dir}/")
    logger.info(f"Log file: {os.path.join(args.output_dir, 'eval.log')}")
    logger.info("=" * 60)
    logger.info("Grounding Evaluation Complete")
    logger.info("=" * 60)
    return metrics


async def run_referring_evaluation(args):
    """Run referring evaluation."""
    logger.info("=" * 60)
    logger.info("HOI Referring Evaluation")
    logger.info("=" * 60)
    logger.info(f"Annotation file: {args.ann_file}")
    logger.info(f"Image prefix: {args.img_prefix}")
    logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info("=" * 60)
    
    # Load annotations
    samples = load_referring_annotations(args.ann_file)
    if args.max_images:
        samples = samples[:args.max_images]
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Create evaluator with lower max_tokens for referring (short action phrases)
    evaluator = HOIAgentEvaluator(
        endpoint=args.endpoint,
        model_name=args.model,
        max_turns=args.max_turns,
        max_tokens=256,  # Referring task outputs short action phrases
        verbose=args.verbose,
        save_thinking=args.save_thinking
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    results, thinking_logs = await evaluate_referring_batch(
        evaluator, samples, args.img_prefix,
        concurrency=args.concurrency,
        save_thinking=args.save_thinking,
        save_visualizations=args.save_viz,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Compute metrics
    metrics = compute_referring_metrics(results, bertscore_gpu=args.bertscore_gpu)
    
    # Compute tool usage stats
    tool_stats = compute_tool_usage_stats(thinking_logs)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    for key, value in sorted(metrics.items()):
        logger.info(f"{key}: {value:.4f}")
    logger.info("=" * 60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'per_sample_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'tool_usage_stats.json'), 'w') as f:
        json.dump(tool_stats, f, indent=2)
    
    if args.save_thinking and thinking_logs:
        with open(os.path.join(args.output_dir, 'thinking.jsonl'), 'w') as f:
            for log in thinking_logs:
                f.write(json.dumps(asdict(log)) + '\n')
    
    # Per-action stats
    action_stats = defaultdict(lambda: {'count': 0, 'correct': 0})
    for r in results:
        action = r['ground_truth']
        action_stats[action]['count'] += 1
        if r['exact_match']:
            action_stats[action]['correct'] += 1
    
    for action in action_stats:
        action_stats[action]['accuracy'] = \
            action_stats[action]['correct'] / action_stats[action]['count']
    
    with open(os.path.join(args.output_dir, 'action_stats.json'), 'w') as f:
        json.dump(dict(action_stats), f, indent=2)
    
    # W&B logging
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"referring_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
            wandb.log(metrics)
            wandb.log({"tool_stats": tool_stats})
            wandb.finish()
        except ImportError:
            logger.warning("wandb not available")
    
    logger.info(f"\nResults saved to {args.output_dir}/")
    logger.info(f"Log file: {os.path.join(args.output_dir, 'eval.log')}")
    logger.info("=" * 60)
    logger.info("Referring Evaluation Complete")
    logger.info("=" * 60)
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HOI Agent Evaluation with Tool Calling (Multi-GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Task configuration
    parser.add_argument("--task", type=str, required=True,
                        choices=['grounding', 'referring'],
                        help="Task type: grounding or referring")
    parser.add_argument("--dataset", type=str, default="hico",
                        choices=['hico', 'swig'],
                        help="Dataset: hico or swig")
    
    # Data paths
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to simplified annotation JSON file")
    parser.add_argument("--img-prefix", type=str, required=True,
                        help="Prefix path to images directory")
    parser.add_argument("--output-dir", type=str, default="results/hoi_eval",
                        help="Output directory for results")
    
    # Model configuration
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1",
                        help="vLLM server endpoint")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name served by vLLM")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Maximum agent turns (default: 10)")
    
    # Parallelization
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Async concurrency per worker (default: 8)")
    parser.add_argument("--bertscore-gpu", type=int, default=0,
                        help="GPU for BERTScore computation (default: 0)")
    
    # Evaluation options
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum images to evaluate (default: all)")
    parser.add_argument("--save-thinking", action="store_true",
                        help="Save thinking/reasoning logs")
    parser.add_argument("--save-viz", action="store_true",
                        help="Save visualization images (auto-enabled with --verbose)")
    parser.add_argument("--unique-run", action="store_true", default=True,
                        help="Append timestamp to output dir for unique runs")
    parser.add_argument("--no-unique-run", action="store_false", dest="unique_run",
                        help="Disable unique run (overwrite existing results)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    # W&B integration
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="hoi-eval",
                        help="W&B project name")
    
    args = parser.parse_args()
    
    # Apply unique output directory if enabled
    if args.unique_run:
        args.output_dir = get_unique_output_dir(args.output_dir)
    
    # Setup logging BEFORE any other operations
    log_path = setup_logging(args.output_dir)
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("HOI Agent Evaluation - Session Started")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("")
    logger.info("Configuration:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    
    # Auto-enable save_viz when verbose is set
    if args.verbose and not args.save_viz:
        args.save_viz = True
    
    # Set default annotation files if not specified
    if not args.ann_file:
        ann_files = {
            ('hico', 'grounding'): 'data/benchmarks_simplified/hico_ground_test_simplified.json',
            ('hico', 'referring'): 'data/benchmarks_simplified/hico_action_referring_test_simplified.json',
            ('swig', 'grounding'): 'data/benchmarks_simplified/swig_ground_test_simplified.json',
            ('swig', 'referring'): 'data/benchmarks_simplified/swig_action_referring_test_simplified.json',
        }
        args.ann_file = ann_files.get((args.dataset, args.task))
    
    # Set default image prefix if not specified
    if not args.img_prefix:
        img_prefixes = {
            'hico': 'data/hico_20160224_det/images/test2015',
            'swig': 'data/swig_hoi/images_512',
        }
        args.img_prefix = img_prefixes.get(args.dataset)
    
    # Run evaluation
    if args.task == 'grounding':
        asyncio.run(run_grounding_evaluation(args))
    else:
        asyncio.run(run_referring_evaluation(args))


if __name__ == "__main__":
    main()
