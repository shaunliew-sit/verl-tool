#!/usr/bin/env python3
"""
HOI Detection Agent Evaluation with Spatial Linking (HuggingFace-based)

Evaluates a trained HOI model with spatial linking using HuggingFace generation.
This script uses SpatialLinkingInteractionModel for inference with full spatial
linking capabilities, including the cross-attention module.

Key Features:
- Uses SpatialLinkingInteractionModel instead of vLLM
- Loads LoRA adapter and spatial_linking weights
- Full tool-calling support (zoom_in, zoom_out, detect_objects)
- Supports both grounding and referring tasks
- Multi-GPU parallel evaluation using torch.multiprocessing
- Resume capability for interrupted evaluations
- W&B integration for metrics tracking

Usage:
    # Run grounding evaluation (single GPU)
    python examples/eval/hoi/eval_hoi_spatial_agent.py \
        --task grounding \
        --dataset hico \
        --base-model Qwen/Qwen3-VL-8B-Instruct \
        --lora-path checkpoints/actor/lora_adapter \
        --spatial-ckpt checkpoints/actor/spatial_linking_grpo.pt \
        --ann-file data/benchmarks_simplified/hico_ground_test_simplified.json \
        --img-prefix data/hico_20160224_det/images/test2015 \
        --max-images 10 --verbose --wandb

    # Run referring evaluation with multi-GPU (8 GPUs)
    python examples/eval/hoi/eval_hoi_spatial_agent.py \
        --task referring \
        --dataset hico \
        --base-model Qwen/Qwen3-VL-8B-Instruct \
        --lora-path checkpoints/actor/lora_adapter \
        --spatial-ckpt checkpoints/actor/spatial_linking_grpo.pt \
        --ann-file data/benchmarks_simplified/hico_action_referring_test_simplified.json \
        --img-prefix data/hico_20160224_det/images/test2015 \
        --num-gpus 8 --verbose --wandb

    # Resume an interrupted evaluation
    python examples/eval/hoi/eval_hoi_spatial_agent.py \
        --task referring \
        --dataset hico \
        --base-model Qwen/Qwen3-VL-8B-Instruct \
        --lora-path checkpoints/actor/lora_adapter \
        --spatial-ckpt checkpoints/actor/spatial_linking_grpo.pt \
        --ann-file data/benchmarks_simplified/hico_action_referring_test_simplified.json \
        --img-prefix data/hico_20160224_det/images/test2015 \
        --resume results/hoi_spatial_eval_20260124_120000
"""

import os
import sys
import json
import argparse
import re
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add project root and spatial_linking_training to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, "/workspace/spatial_linking_training")
sys.path.insert(0, "/workspace")

# Setup logging
logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, log_filename: str = "eval_spatial.log") -> str:
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)
    
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    return log_path


def get_unique_output_dir(base_dir: str) -> str:
    """Generate unique output directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_dir}_{timestamp}"


def load_previous_results(resume_dir: str) -> Tuple[List[Dict], set]:
    """Load previous results for resuming evaluation.
    
    Args:
        resume_dir: Path to previous output directory
        
    Returns:
        Tuple of (previous_results, completed_file_names)
    """
    results_file = os.path.join(resume_dir, 'per_sample_results.json')
    
    if not os.path.exists(results_file):
        logger.warning(f"No previous results found at {results_file}")
        return [], set()
    
    with open(results_file, 'r') as f:
        previous_results = json.load(f)
    
    completed_files = set()
    for r in previous_results:
        if 'file_name' in r:
            completed_files.add(r['file_name'])
    
    logger.info(f"Loaded {len(previous_results)} previous results from {resume_dir}")
    logger.info(f"Completed files: {len(completed_files)}")
    
    return previous_results, completed_files


# =============================================================================
# Constants
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant for Human-Object Interaction detection.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "zoom_in", "description": "Zoom in on a specific region of the image to examine details.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "Bounding box coordinates [x1, y1, x2, y2] in 1000x1000 normalized format.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to zoom in on. Use 1 for the main image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "zoom_out", "description": "Reset the view to the original full image.", "parameters": {"type": "object", "properties": {"target_image": {"type": "number", "description": "The index of the image to reset. Use 1 for the main image."}}, "required": ["target_image"]}}}
{"type": "function", "function": {"name": "detect_objects", "description": "Detect objects in the image using Grounding DINO.", "parameters": {"type": "object", "properties": {"class_names": {"type": "string", "description": "Classes to detect, separated by ' . ' (e.g., 'person . cup')"}, "target_image": {"type": "number", "description": "The index of the image to detect objects in. Use 1 for the main image."}}, "required": ["class_names", "target_image"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Think in the mind first, and then decide whether to call tools if needed OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."""


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
# Model Loading
# =============================================================================

def load_spatial_model(
    base_model: str,
    lora_path: str,
    spatial_ckpt: str,
    device: str = "cuda:0",
) -> Tuple[Any, Any]:
    """
    Load SpatialLinkingInteractionModel with LoRA adapter and spatial linking weights.
    
    Args:
        base_model: Base model name (e.g., Qwen/Qwen3-VL-8B-Instruct)
        lora_path: Path to LoRA adapter directory
        spatial_ckpt: Path to spatial_linking.pt weights
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    from peft import PeftModel
    from transformers import AutoProcessor
    from spatial_linking_training.models.spatial_model import SpatialLinkingInteractionModel
    
    logger.info(f"Loading SpatialLinkingInteractionModel...")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  LoRA path: {lora_path}")
    logger.info(f"  Spatial checkpoint: {spatial_ckpt}")
    logger.info(f"  Device: {device}")
    
    # Suppress warnings
    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*Some weights.*were not initialized.*")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load base SpatialLinkingInteractionModel
    logger.info("  Loading base model...")
    model = SpatialLinkingInteractionModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    
    # Load LoRA adapter
    if lora_path and os.path.exists(lora_path):
        logger.info(f"  Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        logger.info("  LoRA adapter loaded successfully")
    else:
        logger.warning(f"  LoRA path not found: {lora_path}")
    
    # Load spatial linking weights
    if spatial_ckpt and os.path.exists(spatial_ckpt):
        logger.info(f"  Loading spatial linking weights from {spatial_ckpt}...")
        spatial_state = torch.load(spatial_ckpt, map_location="cpu")
        
        # Handle PEFT-wrapped model structure
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            if hasattr(model.base_model.model, 'spatial_linking'):
                model.base_model.model.spatial_linking.load_state_dict(spatial_state)
                logger.info("  Spatial linking weights loaded (PEFT structure)")
            else:
                logger.warning("  Could not find spatial_linking in PEFT model structure")
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'spatial_linking'):
            model.base_model.spatial_linking.load_state_dict(spatial_state)
            logger.info("  Spatial linking weights loaded")
        elif hasattr(model, 'spatial_linking'):
            model.spatial_linking.load_state_dict(spatial_state)
            logger.info("  Spatial linking weights loaded (direct)")
        else:
            logger.warning("  Could not find spatial_linking module in model")
    else:
        logger.warning(f"  Spatial checkpoint not found: {spatial_ckpt}")
        logger.warning("  Using random initialization for spatial_linking module!")
    
    # Set box token IDs
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        if hasattr(model.base_model.model, 'set_box_token_ids'):
            model.base_model.model.set_box_token_ids(processor.tokenizer)
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'set_box_token_ids'):
        model.base_model.set_box_token_ids(processor.tokenizer)
    elif hasattr(model, 'set_box_token_ids'):
        model.set_box_token_ids(processor.tokenizer)
    
    model.eval()
    logger.info("Model loaded successfully!")
    
    return model, processor


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

Guidelines: Analyze the image to locate human-object interaction pairs. You may use zoom_in to examine details if needed. For each person-object pair performing "{action}", output their bounding boxes in JSON format. Coordinates should be in the 1000x1000 normalized format.

Think in the mind first, and then decide whether to call tools if needed OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."""


def build_referring_prompt(person_box: List[int], object_box: List[int], 
                          object_category: str, width: int, height: int) -> str:
    """Build referring task prompt with normalized coordinates and box tokens for spatial linking."""
    def normalize(box, w, h):
        return [
            int(box[0] * 1000 / w),
            int(box[1] * 1000 / h),
            int(box[2] * 1000 / w),
            int(box[3] * 1000 / h)
        ]
    
    person_norm = normalize(person_box, width, height)
    object_norm = normalize(object_box, width, height)
    
    # Compute interaction box (union of person and object)
    interaction_box = [
        min(person_norm[0], object_norm[0]),
        min(person_norm[1], object_norm[1]),
        max(person_norm[2], object_norm[2]),
        max(person_norm[3], object_norm[3])
    ]
    
    # Format boxes with special tokens for spatial linking
    # The <|box_start|> and <|box_end|> tokens enable spatial linking cross-attention
    person_box_str = f'{{"bbox_2d": {person_norm}, "label": "person"}}'
    object_box_str = f'{{"bbox_2d": {object_norm}, "label": "{object_category}"}}'
    interaction_box_str = f'{{"bbox_2d": {interaction_box}, "label": "interaction"}}'
    
    return (
        f"Question: What action is the person performing with the object?\n"
        f"The first region <|box_start|>{person_box_str}<|box_end|> contains a PERSON. "
        f"The second region <|box_start|>{object_box_str}<|box_end|> contains an OBJECT. "
        f"The interaction region <|box_start|>{interaction_box_str}<|box_end|> shows their interaction.\n"
        f"Respond with ONLY the action phrase in format: \"{{verb}} {{object}}\" "
        f"(e.g., \"riding bicycle\", \"holding cup\"). Use base verb form, no articles.\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> "
        f"(if any tools needed) OR <answer>...</answer> (if no tools needed)."
    )


# =============================================================================
# Spatial Linking Agent Evaluator
# =============================================================================

class SpatialHOIAgentEvaluator:
    """Evaluator that runs the spatial linking model with tool calling."""
    
    def __init__(
        self,
        model: Any,
        processor: Any,
        max_turns: int = 10,
        max_new_tokens: int = 1024,
        verbose: bool = False,
        save_thinking: bool = True,
        enable_grounding_dino: bool = True
    ):
        self.model = model
        self.processor = processor
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.save_thinking = save_thinking
        self.enable_grounding_dino = enable_grounding_dino
        self._grounding_dino_model = None
    
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
    
    def _run_grounding_dino(self, image: Image.Image, query: str, 
                            box_threshold: float = 0.3, text_threshold: float = 0.25) -> List[Dict]:
        """Run Grounding DINO object detection on image."""
        try:
            from groundingdino.util.inference import load_model, load_image, predict
            import tempfile
            
            # Initialize model if not cached
            if self._grounding_dino_model is None:
                import os
                import groundingdino
                
                model_paths = [
                    "/workspace/verl-tool/checkpoints/groundingdino_swint_ogc.pth",
                    os.path.expanduser("~/.cache/groundingdino/groundingdino_swint_ogc.pth"),
                    "/tmp/groundingdino_swint_ogc.pth"
                ]
                
                gd_path = os.path.dirname(groundingdino.__file__)
                config_path = os.path.join(gd_path, "config", "GroundingDINO_SwinT_OGC.py")
                
                weights_path = None
                for path in model_paths:
                    if os.path.exists(path):
                        weights_path = path
                        break
                
                if weights_path is None:
                    import urllib.request
                    weights_path = "/tmp/groundingdino_swint_ogc.pth"
                    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                    logger.info(f"Downloading Grounding DINO weights to {weights_path}...")
                    urllib.request.urlretrieve(url, weights_path)
                
                self._grounding_dino_model = load_model(config_path, weights_path)
                # Use CPU to avoid conflicts with main model
                self._grounding_dino_model = self._grounding_dino_model.cpu()
                logger.info("Grounding DINO model loaded (CPU mode)")
            
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                image.save(f, format='JPEG')
                temp_path = f.name
            
            try:
                import os as os_module
                image_source, image_tensor = load_image(temp_path)
                image_tensor = image_tensor.cpu()
                
                boxes, logits, phrases = predict(
                    model=self._grounding_dino_model,
                    image=image_tensor,
                    caption=query,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device="cpu"
                )
                
                detections = []
                h, w = image_source.shape[:2]
                for box, logit, phrase in zip(boxes, logits, phrases):
                    cx, cy, bw, bh = box.tolist()
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    detections.append({
                        'label': phrase,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(logit)
                    })
                
                return detections
            finally:
                os_module.unlink(temp_path)
                
        except ImportError as e:
            logger.warning(f"Grounding DINO not available: {e}")
            return []
        except Exception as e:
            logger.warning(f"Detection error: {e}")
            return []
    
    def execute_tool(self, tool_name: str, arguments: Dict, state: AgentState) -> str:
        """Execute a tool and return the result."""
        if tool_name == 'zoom_in':
            bbox = arguments.get('bbox', arguments.get('bbox_2d', []))
            if len(bbox) == 4:
                x1_norm, y1_norm, x2_norm, y2_norm = [float(c) for c in bbox]
                w, h = state.current_image.size
                x1 = int(x1_norm * w / 1000)
                y1 = int(y1_norm * h / 1000)
                x2 = int(x2_norm * w / 1000)
                y2 = int(y2_norm * h / 1000)
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
            
            if not self.enable_grounding_dino:
                return f"Detection for '{query}': Grounding DINO disabled."
            
            try:
                detections = self._run_grounding_dino(state.current_image, query)
                if detections:
                    # Format detections for model
                    det_strs = []
                    for det in detections:
                        bbox = det['bbox']
                        # Convert to 1000x1000 normalized format
                        w, h = state.current_image.size
                        norm_bbox = [
                            int(bbox[0] * 1000 / w),
                            int(bbox[1] * 1000 / h),
                            int(bbox[2] * 1000 / w),
                            int(bbox[3] * 1000 / h)
                        ]
                        det_strs.append(
                            f'{{"bbox_2d": {norm_bbox}, "label": "{det["label"]}", "confidence": {det["confidence"]:.2f}}}'
                        )
                    return f"Detected objects:\n" + "\n".join(det_strs)
                else:
                    return f"No objects detected for query '{query}'."
            except Exception as e:
                return f"Detection failed: {e}"
        
        return f"Unknown tool: {tool_name}"
    
    def generate_response(self, messages: List[Dict], image: Image.Image,
                          refer_boxes: Optional[torch.Tensor] = None) -> str:
        """Generate a response using the HuggingFace model.
        
        Args:
            messages: List of conversation messages
            image: PIL Image to process
            refer_boxes: Optional tensor of shape [N, 4] with bounding boxes in 1000x1000 format
                        for spatial linking enhancement
        """
        # Build chat text
        chat_text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                chat_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                chat_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                chat_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        chat_text += "<|im_start|>assistant\n"
        
        # Process image and text
        inputs = self.processor(
            text=chat_text,
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Prepare generation kwargs
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # Add refer_boxes for spatial linking if provided
        if refer_boxes is not None:
            # refer_boxes should be [N, 4] tensor, wrap in list for batch dimension
            refer_boxes_device = refer_boxes.to(device)
            generate_kwargs["refer_boxes"] = [refer_boxes_device]
            if self.verbose:
                logger.debug(f"Spatial linking enabled with {refer_boxes.shape[0]} boxes")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)
        
        # Decode only the generated tokens
        input_len = inputs['input_ids'].shape[1]
        generated = outputs[0][input_len:]
        response = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response.strip()
    
    def run_agent_loop(self, image_path: str, prompt: str, 
                       refer_boxes: Optional[torch.Tensor] = None) -> Dict:
        """Run the agent loop for a single sample.
        
        Args:
            image_path: Path to the image file
            prompt: The user prompt
            refer_boxes: Optional tensor of shape [N, 4] with bounding boxes in 1000x1000 format
                        for spatial linking (typically [person_box, object_box, interaction_box])
        """
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
                content = self.generate_response(state.messages, state.current_image, 
                                                  refer_boxes=refer_boxes)
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error generating response: {e}")
                break
            
            # Extract thinking blocks
            thinking = self._extract_thinking_blocks(content)
            state.thinking_blocks.extend(thinking)
            
            # Check for tool calls
            tool_calls = self._extract_tool_calls_from_content(content)
            
            if tool_calls:
                state.messages.append({
                    'role': 'assistant',
                    'content': content
                })
                
                # Execute tools
                tool_results = []
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
                    tool_results.append(f"[{tool_name}] {result}")
                
                tool_output = "\n".join(tool_results)
                state.messages.append({
                    'role': 'user',
                    'content': f"Tool output:\n{tool_output}\n\nContinue your analysis. If you have enough information, provide your final answer. Format: <think>...</think> <answer>...</answer>"
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


def extract_boxes_from_response(response: str, max_boxes: int = 20) -> List[Dict]:
    """Extract bounding boxes from model response."""
    boxes = []
    
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    search_text = answer_match.group(1) if answer_match else response
    
    pattern = r'\{\s*"bbox_2d"\s*:\s*\[([^\]]+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, search_text)
    
    for coords_str, label in matches:
        if len(boxes) >= max_boxes:
            break
        try:
            coords = [int(float(x.strip())) for x in coords_str.split(',')]
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1100 and y2 <= 1100:
                    boxes.append({"bbox_2d": coords, "label": label.lower()})
        except:
            continue
    
    if not boxes:
        bbox_pattern = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
        bbox_matches = re.findall(bbox_pattern, search_text)
        for i, (x1, y1, x2, y2) in enumerate(bbox_matches):
            if len(boxes) >= max_boxes:
                break
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1100 and y2 <= 1100:
                label = "person" if i % 2 == 0 else "object"
                boxes.append({"bbox_2d": [x1, y1, x2, y2], "label": label})
    
    return boxes


def extract_pairs_from_boxes(boxes: List[Dict]) -> List[Tuple[List[int], List[int]]]:
    """Extract person-object pairs from boxes."""
    persons = [b['bbox_2d'] for b in boxes if 'person' in b['label'].lower()]
    objects = [b['bbox_2d'] for b in boxes if 'person' not in b['label'].lower()]
    
    pairs = []
    for p in persons:
        for o in objects:
            pairs.append((p, o))
    
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


def clean_action_text(text: str) -> str:
    """Clean action text for comparison."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = ' '.join(text.split())
    return text.strip()


def extract_action_from_response(response: str) -> str:
    """Extract action phrase from model response."""
    response = clean_action_text(response)
    
    # Look for answer tags
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip().rstrip('.!?,;:')
    
    # Look for "Final Answer:" pattern
    match = re.search(r'final\s*answer[:\.\s]+\s*([^\n]+)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip('.!?,;:')
    
    # Look for "action:" pattern
    match = re.search(r'action[:\.\s]+\s*([^\n]+)', response, re.IGNORECASE)
    if match:
        result = match.group(1).strip()
        if result and result not in ['phrase is', 'is', 'phrase']:
            return result.rstrip('.!?,;:')
    
    return response[:100].rstrip('.!?,;:') if response else ""


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_grounding_batch(
    evaluator: SpatialHOIAgentEvaluator,
    samples: List[GroundingSample],
    img_prefix: str,
    save_thinking: bool = True,
    verbose: bool = False
) -> Tuple[List[Dict], List[ThinkingLog]]:
    """Evaluate grounding task on a batch of samples."""
    results = []
    thinking_logs = []
    
    for idx, sample in enumerate(tqdm(samples, desc="Grounding")):
        image_path = os.path.join(img_prefix, sample.file_name)
        
        if not os.path.exists(image_path):
            if verbose:
                logger.warning(f"Image not found: {image_path}")
            continue
        
        # Build ground truth pairs
        gt_pairs = []
        for i in range(sample.num_pairs):
            person_idx = sample.gt_box_inds[i * 2]
            object_idx = sample.gt_box_inds[i * 2 + 1]
            gt_pairs.append((sample.boxes[person_idx], sample.boxes[object_idx]))
        
        prompt = build_grounding_prompt(sample.action, sample.object_category)
        
        try:
            result = evaluator.run_agent_loop(image_path, prompt)
        except Exception as e:
            if verbose:
                logger.error(f"Error processing {sample.file_name}: {e}")
            continue
        
        # Extract predictions
        pred_boxes = extract_boxes_from_response(result['response'])
        
        # Convert coordinates if needed
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()
            
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
        results.append(result_entry)
        
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
            thinking_logs.append(thinking_log)
        
        if verbose:
            logger.info(f"Sample {idx}: recall@0.5={sample_metrics.get('recall@0.5', 0):.2f}")
    
    return results, thinking_logs


def normalize_box_to_1000(box: List[int], width: int, height: int) -> List[int]:
    """Normalize box from pixel coordinates to 1000x1000 format."""
    return [
        int(box[0] * 1000 / width),
        int(box[1] * 1000 / height),
        int(box[2] * 1000 / width),
        int(box[3] * 1000 / height)
    ]


def compute_interaction_box(person_box: List[int], object_box: List[int]) -> List[int]:
    """Compute interaction box as the union of person and object boxes."""
    return [
        min(person_box[0], object_box[0]),
        min(person_box[1], object_box[1]),
        max(person_box[2], object_box[2]),
        max(person_box[3], object_box[3])
    ]


def evaluate_referring_batch(
    evaluator: SpatialHOIAgentEvaluator,
    samples: List[ReferringSample],
    img_prefix: str,
    save_thinking: bool = True,
    verbose: bool = False
) -> Tuple[List[Dict], List[ThinkingLog]]:
    """Evaluate referring task on a batch of samples."""
    results = []
    thinking_logs = []
    
    for idx, sample in enumerate(tqdm(samples, desc="Referring")):
        image_path = os.path.join(img_prefix, sample.file_name)
        
        if not os.path.exists(image_path):
            if verbose:
                logger.warning(f"Image not found: {image_path}")
            continue
        
        person_box = sample.boxes[sample.person_box_idx]
        object_box = sample.boxes[sample.object_box_idx]
        
        # Normalize boxes to 1000x1000 format for spatial linking
        person_box_norm = normalize_box_to_1000(person_box, sample.width, sample.height)
        object_box_norm = normalize_box_to_1000(object_box, sample.width, sample.height)
        interaction_box = compute_interaction_box(person_box_norm, object_box_norm)
        
        # Create refer_boxes tensor for spatial linking [3, 4]
        refer_boxes_tensor = torch.tensor(
            [person_box_norm, object_box_norm, interaction_box],
            dtype=torch.float32
        )
        
        words = sample.gt_action.split()
        object_category = words[-1] if words else "object"
        
        prompt = build_referring_prompt(
            person_box, object_box, object_category,
            sample.width, sample.height
        )
        
        try:
            result = evaluator.run_agent_loop(image_path, prompt, refer_boxes=refer_boxes_tensor)
        except Exception as e:
            if verbose:
                logger.error(f"Error processing {sample.file_name}: {e}")
            continue
        
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
        results.append(result_entry)
        
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
            thinking_logs.append(thinking_log)
        
        if verbose:
            logger.info(f"Sample {idx}: GT='{gt}', Pred='{pred_clean}', Match={exact_match}")
    
    return results, thinking_logs


# =============================================================================
# Multi-GPU Worker Functions
# =============================================================================

def _worker_init(gpu_id: int, args_dict: Dict) -> Tuple[Any, Any, Any]:
    """Initialize worker with model on specific GPU.
    
    This function is called once per worker process to load the model.
    When CUDA_VISIBLE_DEVICES is set by parent (e.g., "4,5,6,7"), 
    gpu_id 0 maps to physical GPU 4, gpu_id 1 maps to physical GPU 5, etc.
    """
    import torch
    import warnings
    warnings.filterwarnings("ignore")
    
    # DO NOT override CUDA_VISIBLE_DEVICES - respect parent's setting
    # Just use the gpu_id as the cuda device index within visible devices
    
    # Reimport to ensure fresh state
    from peft import PeftModel
    from transformers import AutoProcessor
    from spatial_linking_training.models.spatial_model import SpatialLinkingInteractionModel
    
    # gpu_id is the index within visible GPUs (0, 1, 2, 3 -> maps to physical 4, 5, 6, 7)
    device = f"cuda:{gpu_id}"
    
    # Log which GPU we're using
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"[Worker {gpu_id}] Loading model on {device} (CUDA_VISIBLE_DEVICES={visible_devices})...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args_dict['base_model'], trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load base model
    model = SpatialLinkingInteractionModel.from_pretrained(
        args_dict['base_model'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    
    # Load LoRA adapter
    if args_dict.get('lora_path') and os.path.exists(args_dict['lora_path']):
        model = PeftModel.from_pretrained(model, args_dict['lora_path'])
        print(f"[Worker {gpu_id}] LoRA adapter loaded")
    
    # Load spatial linking weights
    spatial_ckpt = args_dict.get('spatial_ckpt')
    if spatial_ckpt and os.path.exists(spatial_ckpt):
        spatial_state = torch.load(spatial_ckpt, map_location="cpu")
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            if hasattr(model.base_model.model, 'spatial_linking'):
                model.base_model.model.spatial_linking.load_state_dict(spatial_state)
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'spatial_linking'):
            model.base_model.spatial_linking.load_state_dict(spatial_state)
        elif hasattr(model, 'spatial_linking'):
            model.spatial_linking.load_state_dict(spatial_state)
        print(f"[Worker {gpu_id}] Spatial linking weights loaded")
    
    # Set box token IDs
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        if hasattr(model.base_model.model, 'set_box_token_ids'):
            model.base_model.model.set_box_token_ids(processor.tokenizer)
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'set_box_token_ids'):
        model.base_model.set_box_token_ids(processor.tokenizer)
    elif hasattr(model, 'set_box_token_ids'):
        model.set_box_token_ids(processor.tokenizer)
    
    model.eval()
    print(f"[Worker {gpu_id}] Model ready on {device}")
    
    # Create evaluator
    evaluator = SpatialHOIAgentEvaluator(
        model=model,
        processor=processor,
        max_turns=args_dict.get('max_turns', 5),
        verbose=args_dict.get('verbose', False),
        save_thinking=args_dict.get('save_thinking', True),
        enable_grounding_dino=args_dict.get('enable_grounding_dino', True)
    )
    
    return evaluator, model, processor


def worker_evaluate_grounding(
    gpu_id: int,
    samples_data: List[Dict],
    args_dict: Dict
) -> Tuple[List[Dict], List[Dict]]:
    """Worker function to evaluate grounding on a specific GPU.
    
    Args:
        gpu_id: The GPU ID to use for this worker
        samples_data: List of sample dictionaries (serializable)
        args_dict: Arguments dictionary (serializable)
        
    Returns:
        Tuple of (results, thinking_logs) as serializable dicts
    """
    # Initialize worker
    evaluator, model, processor = _worker_init(gpu_id, args_dict)
    
    # Convert samples_data back to GroundingSample objects
    samples = []
    for d in samples_data:
        samples.append(GroundingSample(**d))
    
    # Run evaluation
    results, thinking_logs = evaluate_grounding_batch(
        evaluator, samples, args_dict['img_prefix'],
        save_thinking=args_dict.get('save_thinking', True),
        verbose=args_dict.get('verbose', False)
    )
    
    # Convert thinking logs to dicts for serialization
    thinking_logs_dicts = [asdict(log) for log in thinking_logs]
    
    print(f"[Worker {gpu_id}] Completed {len(results)} samples")
    
    return results, thinking_logs_dicts


def worker_evaluate_referring(
    gpu_id: int,
    samples_data: List[Dict],
    args_dict: Dict
) -> Tuple[List[Dict], List[Dict]]:
    """Worker function to evaluate referring on a specific GPU.
    
    Args:
        gpu_id: The GPU ID to use for this worker
        samples_data: List of sample dictionaries (serializable)
        args_dict: Arguments dictionary (serializable)
        
    Returns:
        Tuple of (results, thinking_logs) as serializable dicts
    """
    # Initialize worker
    evaluator, model, processor = _worker_init(gpu_id, args_dict)
    
    # Convert samples_data back to ReferringSample objects
    samples = []
    for d in samples_data:
        samples.append(ReferringSample(**d))
    
    # Run evaluation
    results, thinking_logs = evaluate_referring_batch(
        evaluator, samples, args_dict['img_prefix'],
        save_thinking=args_dict.get('save_thinking', True),
        verbose=args_dict.get('verbose', False)
    )
    
    # Convert thinking logs to dicts for serialization
    thinking_logs_dicts = [asdict(log) for log in thinking_logs]
    
    print(f"[Worker {gpu_id}] Completed {len(results)} samples")
    
    return results, thinking_logs_dicts


def run_multi_gpu_evaluation(
    task: str,
    samples: List[Any],
    args: argparse.Namespace,
    num_gpus: int
) -> Tuple[List[Dict], List[ThinkingLog]]:
    """Run evaluation in parallel across multiple GPUs.
    
    Args:
        task: 'grounding' or 'referring'
        samples: List of sample objects
        args: Parsed arguments
        num_gpus: Number of GPUs to use
        
    Returns:
        Tuple of (all_results, all_thinking_logs)
    """
    # Convert samples to serializable dicts
    if task == 'grounding':
        samples_data = [asdict(s) for s in samples]
        worker_fn = worker_evaluate_grounding
    else:
        samples_data = [asdict(s) for s in samples]
        worker_fn = worker_evaluate_referring
    
    # Convert args to dict for serialization
    args_dict = {
        'base_model': args.base_model,
        'lora_path': args.lora_path,
        'spatial_ckpt': args.spatial_ckpt,
        'img_prefix': args.img_prefix,
        'max_turns': args.max_turns,
        'verbose': args.verbose,
        'save_thinking': args.save_thinking,
        'enable_grounding_dino': args.enable_grounding_dino,
    }
    
    # Split samples across GPUs
    chunk_size = (len(samples_data) + num_gpus - 1) // num_gpus
    chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(samples_data))
        if start_idx < len(samples_data):
            chunks.append(samples_data[start_idx:end_idx])
    
    actual_num_gpus = len(chunks)
    logger.info(f"Distributing {len(samples_data)} samples across {actual_num_gpus} GPUs")
    for i, chunk in enumerate(chunks):
        logger.info(f"  GPU {i}: {len(chunk)} samples")
    
    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Create process pool and run workers
    all_results = []
    all_thinking_logs_dicts = []
    
    with mp.Pool(processes=actual_num_gpus) as pool:
        # Prepare worker arguments
        worker_args = [
            (gpu_id, chunks[gpu_id], args_dict)
            for gpu_id in range(actual_num_gpus)
        ]
        
        # Run in parallel
        logger.info("Starting parallel evaluation...")
        results_list = pool.starmap(worker_fn, worker_args)
        
        # Collect results
        for results, thinking_logs_dicts in results_list:
            all_results.extend(results)
            all_thinking_logs_dicts.extend(thinking_logs_dicts)
    
    # Convert thinking logs back to ThinkingLog objects
    all_thinking_logs = []
    for log_dict in all_thinking_logs_dicts:
        all_thinking_logs.append(ThinkingLog(**log_dict))
    
    logger.info(f"Multi-GPU evaluation complete: {len(all_results)} results collected")
    
    return all_results, all_thinking_logs


def get_box_area(box: List[int]) -> float:
    """Get box area."""
    return (box[2] - box[0]) * (box[3] - box[1])


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
    
    metrics['num_samples'] = len(predictions)
    
    return metrics


def compute_referring_metrics(predictions: List[Dict], bertscore_gpu: int = 0) -> Dict:
    """Compute referring metrics (Exact Match, METEOR, CIDEr, BERTScore)."""
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
    metrics['num_samples'] = len(predictions)
    
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
        logger.info(f"METEOR score computed: {metrics['meteor']:.4f}")
    except ImportError:
        logger.warning("NLTK not available for METEOR score")
    except Exception as e:
        logger.warning(f"METEOR computation failed: {e}")
    
    # Try CIDEr
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # Format for CIDEr
        gts = {i: [clean_action_text(gt)] for i, gt in enumerate(gt_texts)}
        res = {i: [clean_action_text(pred)] for i, pred in enumerate(pred_texts)}
        
        cider = Cider()
        score, _ = cider.compute_score(gts, res)
        metrics['cider'] = score
        logger.info(f"CIDEr score computed: {metrics['cider']:.4f}")
    except ImportError:
        logger.warning("pycocoevalcap not available for CIDEr score")
    except Exception as e:
        logger.warning(f"CIDEr computation failed: {e}")
    
    # Try BERTScore with microsoft/deberta-v2-xxlarge-mnli
    try:
        import os as os_mod
        os_mod.environ["CUDA_VISIBLE_DEVICES"] = str(bertscore_gpu)
        from bert_score import score as bert_score
        
        pred_clean = [clean_action_text(p) for p in pred_texts]
        gt_clean = [clean_action_text(g) for g in gt_texts]
        
        # Filter out empty strings
        valid_pairs = [(p, g) for p, g in zip(pred_clean, gt_clean) if p and g]
        
        if valid_pairs:
            preds, refs = zip(*valid_pairs)
            logger.info("Computing BERTScore with microsoft/deberta-v2-xxlarge-mnli...")
            P, R, F1 = bert_score(
                list(preds), list(refs),
                model_type="microsoft/deberta-v2-xxlarge-mnli",
                lang="en",
                batch_size=32,
                rescale_with_baseline=False,
                verbose=False
            )
            
            metrics['bertscore_precision'] = float(P.mean())
            metrics['bertscore_recall'] = float(R.mean())
            metrics['bertscore_f1'] = float(F1.mean())
            logger.info(f"BERTScore computed: P={metrics['bertscore_precision']:.4f}, R={metrics['bertscore_recall']:.4f}, F1={metrics['bertscore_f1']:.4f}")
    except ImportError:
        logger.warning("bert_score not available")
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
    
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HOI Spatial Linking Agent Evaluation (HuggingFace-based)",
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
    parser.add_argument("--output-dir", type=str, default="results/hoi_spatial_eval",
                        help="Output directory for results")
    
    # Model configuration
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base model name")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--spatial-ckpt", type=str, required=True,
                        help="Path to spatial_linking.pt weights")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    parser.add_argument("--max-turns", type=int, default=5,
                        help="Maximum agent turns (default: 5)")
    
    # Evaluation options
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum images to evaluate (default: all)")
    parser.add_argument("--save-thinking", action="store_true",
                        help="Save thinking/reasoning logs")
    parser.add_argument("--unique-run", action="store_true", default=True,
                        help="Append timestamp to output dir for unique runs")
    parser.add_argument("--no-unique-run", action="store_false", dest="unique_run",
                        help="Disable unique run")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    # Tool options
    parser.add_argument("--enable-grounding-dino", action="store_true", default=True,
                        help="Enable Grounding DINO for object detection tool")
    parser.add_argument("--disable-grounding-dino", action="store_false", dest="enable_grounding_dino",
                        help="Disable Grounding DINO (return placeholder for detect_objects)")
    
    # Metrics options
    parser.add_argument("--bertscore-gpu", type=int, default=0,
                        help="GPU for BERTScore computation (default: 0)")
    
    # Multi-GPU options
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use for parallel evaluation (default: 1)")
    
    # W&B integration
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="hoi-spatial-eval",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")
    
    # Resume options
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to previous output directory to resume from")
    
    args = parser.parse_args()
    
    # Handle resume mode
    previous_results = []
    completed_files = set()
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Resume directory not found: {args.resume}")
            sys.exit(1)
        # Use the resume directory as output directory
        args.output_dir = args.resume
        args.unique_run = False  # Don't generate new directory
        previous_results, completed_files = load_previous_results(args.resume)
    elif args.unique_run:
        # Apply unique output directory if enabled and not resuming
        args.output_dir = get_unique_output_dir(args.output_dir)
    
    # Setup logging
    log_path = setup_logging(args.output_dir)
    
    logger.info("=" * 80)
    logger.info("HOI Spatial Linking Agent Evaluation - Session Started")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("")
    logger.info("Configuration:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    
    # For single-GPU mode, load model in main process
    # For multi-GPU mode, each worker loads its own model
    model, processor, evaluator = None, None, None
    if args.num_gpus == 1:
        # Load model
        model, processor = load_spatial_model(
            base_model=args.base_model,
            lora_path=args.lora_path,
            spatial_ckpt=args.spatial_ckpt,
            device=args.device,
        )
        
        # Create evaluator
        evaluator = SpatialHOIAgentEvaluator(
            model=model,
            processor=processor,
            max_turns=args.max_turns,
            verbose=args.verbose,
            save_thinking=args.save_thinking,
            enable_grounding_dino=args.enable_grounding_dino
        )
    else:
        logger.info(f"Multi-GPU mode: Model will be loaded by each worker process")
    
    # Load data
    if args.task == 'grounding':
        samples = load_grounding_annotations(args.ann_file)
    else:
        samples = load_referring_annotations(args.ann_file)
    
    if args.max_images:
        samples = samples[:args.max_images]
    
    total_samples = len(samples)
    logger.info(f"Loaded {total_samples} samples")
    
    # Filter out already completed samples if resuming
    if completed_files:
        samples = [s for s in samples if s.file_name not in completed_files]
        logger.info(f"Resuming: {len(completed_files)} already completed, {len(samples)} remaining")
    
    # Run evaluation (skip if no samples remaining)
    results = []
    thinking_logs = []
    
    if len(samples) == 0:
        logger.info("No remaining samples to evaluate - all samples already completed")
    elif args.num_gpus > 1:
        # Multi-GPU evaluation
        logger.info(f"Using multi-GPU evaluation with {args.num_gpus} GPUs")
        results, thinking_logs = run_multi_gpu_evaluation(
            task=args.task,
            samples=samples,
            args=args,
            num_gpus=args.num_gpus
        )
    else:
        # Single-GPU evaluation (original path)
        if args.task == 'grounding':
            results, thinking_logs = evaluate_grounding_batch(
                evaluator, samples, args.img_prefix,
                save_thinking=args.save_thinking,
                verbose=args.verbose
            )
        else:
            results, thinking_logs = evaluate_referring_batch(
                evaluator, samples, args.img_prefix,
                save_thinking=args.save_thinking,
                verbose=args.verbose
            )
    
    # Merge with previous results if resuming
    if previous_results:
        logger.info(f"Merging {len(results)} new results with {len(previous_results)} previous results")
        results = previous_results + results
        logger.info(f"Total results: {len(results)}")
    
    # Compute metrics
    if args.task == 'grounding':
        metrics = compute_grounding_metrics(results)
    else:
        metrics = compute_referring_metrics(results, bertscore_gpu=args.bertscore_gpu)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Convert tuples to lists for JSON serialization
    serializable_results = []
    for r in results:
        r_copy = r.copy()
        if 'gt_pairs' in r_copy:
            r_copy['gt_pairs'] = [[list(p), list(o)] for p, o in r['gt_pairs']]
        if 'pred_pairs' in r_copy:
            r_copy['pred_pairs'] = [[list(p), list(o)] for p, o in r['pred_pairs']]
        serializable_results.append(r_copy)
    
    with open(os.path.join(args.output_dir, 'per_sample_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    if args.save_thinking and thinking_logs:
        with open(os.path.join(args.output_dir, 'thinking.jsonl'), 'w') as f:
            for log in thinking_logs:
                f.write(json.dumps(asdict(log)) + '\n')
    
    # Compute tool usage statistics
    tool_stats = {
        'total_samples': len(results),
        'samples_with_tools': sum(1 for r in results if r.get('num_tool_calls', 0) > 0),
        'avg_turns': np.mean([r.get('num_turns', 1) for r in results]) if results else 0,
        'avg_tool_calls': np.mean([r.get('num_tool_calls', 0) for r in results]) if results else 0,
    }
    
    # W&B logging
    if args.wandb:
        try:
            import wandb
            
            run_name = args.wandb_run_name
            if run_name is None:
                run_name = f"{args.task}_{args.dataset}_spatial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args)
            )
            wandb.log(metrics)
            wandb.log({"tool_stats": tool_stats})
            
            # Log summary table
            wandb.log({
                "task": args.task,
                "dataset": args.dataset,
                "num_samples": len(results),
                "num_gpus": args.num_gpus,
            })
            
            wandb.finish()
            logger.info(f"W&B logging complete: {args.wandb_project}/{run_name}")
        except ImportError:
            logger.warning("wandb not available - skipping W&B logging")
        except Exception as e:
            logger.warning(f"W&B logging failed: {e}")
    
    logger.info(f"\nResults saved to {args.output_dir}/")
    logger.info("=" * 60)
    logger.info("Spatial Linking Evaluation Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
