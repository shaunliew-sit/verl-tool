#!/usr/bin/env python3
"""
HOI Detection Agent Evaluation with Tool Calling

Evaluates a trained HOI model with full tool-calling capability.
Uses vLLM server for inference and the HOI detector tools.

Usage:
    # First, start vLLM server with the trained model:
    python -m vllm.entrypoints.openai.api_server \
        --model checkpoints/.../global_step_100/actor/huggingface \
        --served-model-name hoi-trained \
        --port 8000 \
        --trust-remote-code

    # Then run evaluation
    # Interactive mode
    uv run python examples/eval/hoi/eval_hoi_agent.py \
    --endpoint http://localhost:8000/v1 --model hoi-trained --interactive \
    --image-path <IMAGE> \
    --person-bbox "[x1,y1,x2,y2]" --object-bbox "[x1,y1,x2,y2]" \
    --object-label <LABEL> --ground-truth "<ACTION>"
    
    uv run python examples/eval/hoi/eval_hoi_agent.py \
    --endpoint http://localhost:8000/v1 --model hoi-trained --interactive \
    --image-path <IMAGE> \
    --action "<ACTION>" --object-label <LABEL> \
    --ground-truth '[{"bbox_2d":[...],"label":"person"},...]' \
    --output-viz /tmp/viz.jpg
"""

import os
import sys
import json
import argparse
import asyncio
import aiohttp
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import HOI tools
from verl_tool.servers.tools.hoi_detector import HOIDetectorTool


# Tool definitions for the model
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "zoom_in",
            "description": "Zoom into a specific region of the image to examine details. Useful for seeing hand positions, object details, or interaction points more clearly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Bounding box [x1, y1, x2, y2] in pixel coordinates to zoom into"
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
            "description": "Zoom out to see the full image. Use after zooming in to restore global context.",
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
            "description": "Detect objects in the current image view using Grounding DINO. Returns bounding boxes and labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Object categories to detect, separated by ' . ' (e.g., 'person . bicycle . dog')"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


@dataclass
class AgentState:
    """State for agent conversation."""
    messages: List[Dict]
    current_image: Image.Image
    original_image: Image.Image
    tool_calls: List[Dict]
    zoom_history: List[List[int]]


class HOIAgentEvaluator:
    """Evaluator that runs the model with tool calling."""
    
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        max_turns: int = 5,
        verbose: bool = False
    ):
        self.endpoint = endpoint.rstrip('/')
        self.model_name = model_name
        self.max_turns = max_turns
        self.verbose = verbose
        try:
            self.hoi_tool = HOIDetectorTool()
        except:
            self.hoi_tool = None  # Tool not needed if just testing model responses
        
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string."""
        buffered = BytesIO()
        # Resize if too large to avoid API limits
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    async def call_model(self, messages: List[Dict], images: List[Image.Image] = None) -> Dict:
        """Call the vLLM server with messages and optional images."""
        # Format messages for the API
        formatted_messages = []
        images_added = False
        
        for msg in messages:
            if msg['role'] == 'system':
                formatted_messages.append({
                    'role': 'system',
                    'content': msg['content']
                })
            elif msg['role'] == 'user':
                content = msg['content']
                
                # Add images to the FIRST user message
                if images and not images_added and not isinstance(content, list):
                    content_parts = []
                    # Add images first
                    for img in images:
                        img_base64 = self._encode_image(img)
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        })
                    # Add text after images
                    content_parts.append({"type": "text", "text": content})
                    content = content_parts
                    images_added = True
                
                formatted_messages.append({
                    'role': 'user',
                    'content': content
                })
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
            'max_tokens': 2048,
            'temperature': 0.7,
            'repetition_penalty': 1.1,  # Reduce repetition
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                return await response.json()
    
    def execute_tool(self, tool_name: str, arguments: Dict, state: AgentState) -> str:
        """Execute a tool and return the result."""
        if self.verbose:
            print(f"    [TOOL] {tool_name}({arguments})")
        
        if tool_name == 'zoom_in':
            bbox = arguments.get('bbox', [])
            if len(bbox) == 4:
                # Crop the image
                x1, y1, x2, y2 = [int(c) for c in bbox]
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
            query = arguments.get('query', 'person . object')
            # Use the HOI tool's detection capability
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
    
    async def run_agent_loop(
        self,
        image_path: str,
        prompt: str,
        ground_truth: Any = None
    ) -> Dict:
        """Run the agent loop for a single sample."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Initialize state
        state = AgentState(
            messages=[],
            current_image=image.copy(),
            original_image=image.copy(),
            tool_calls=[],
            zoom_history=[]
        )
        
        # Build initial messages - match training format exactly
        system_prompt = """You are a helpful assistant for Human-Object Interaction detection.

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
        
        state.messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        
        # Run agent loop
        final_response = ""
        for turn in range(self.max_turns):
            if self.verbose:
                print(f"  Turn {turn + 1}/{self.max_turns}")
            
            try:
                response = await self.call_model(state.messages, [state.current_image])
            except Exception as e:
                if self.verbose:
                    print(f"    Error: {e}")
                break
            
            choice = response.get('choices', [{}])[0]
            message = choice.get('message', {})
            content = message.get('content', '')
            
            # Check for tool calls (OpenAI format from vLLM)
            tool_calls = message.get('tool_calls', [])
            
            # Also check for Hermes XML-style tool calls in content
            # (model might produce <tool_call>...</tool_call> in text)
            if not tool_calls and content:
                import re
                xml_tool_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
                xml_matches = re.findall(xml_tool_pattern, content, re.DOTALL)
                if xml_matches:
                    for match in xml_matches:
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
            
            # Detect repetition in response (model getting stuck in a loop)
            # Only trigger if we see the EXACT same phrase repeated many times
            if content and len(content) > 500:
                # Check for exact phrase repetition (like "I will use zoom_in" repeated 10+ times)
                import re
                # Find phrases that repeat more than 5 times consecutively
                repeat_pattern = r'(.{20,}?)\1{5,}'  # Same 20+ char phrase repeated 5+ times
                if re.search(repeat_pattern, content):
                    if self.verbose:
                        print(f"    [WARN] Detected repetitive loop, stopping early")
                    # Extract the non-repetitive part
                    lines = content.split('\n')
                    unique_content = []
                    seen = set()
                    for line in lines:
                        line_stripped = line.strip()
                        if line_stripped and line_stripped not in seen:
                            unique_content.append(line)
                            seen.add(line_stripped)
                    final_response = '\n'.join(unique_content[:10])
                    break
            
            if tool_calls:
                # Add assistant message with tool calls
                state.messages.append({
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': tool_calls
                })
                
                # Execute each tool
                for tc in tool_calls:
                    func = tc.get('function', {})
                    tool_name = func.get('name', '')
                    try:
                        arguments = json.loads(func.get('arguments', '{}'))
                    except:
                        arguments = {}
                    
                    result = self.execute_tool(tool_name, arguments, state)
                    state.tool_calls.append({
                        'name': tool_name,
                        'arguments': arguments,
                        'result': result
                    })
                    
                    # Add tool response
                    state.messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.get('id', ''),
                        'content': result
                    })
            else:
                # No tool calls - this is the final response
                final_response = content
                if self.verbose:
                    print(f"    Response: {final_response[:200]}...")
                break
        
        return {
            'response': final_response,
            'tool_calls': state.tool_calls,
            'num_turns': turn + 1,
            'zoom_history': state.zoom_history
        }
    
    def evaluate_grounding(self, response: str, ground_truth: List) -> Dict:
        """Evaluate grounding task."""
        # Extract boxes from response
        import regex as re
        pred_boxes = []
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    if isinstance(item, dict) and 'bbox_2d' in item:
                        pred_boxes.append(item['bbox_2d'])
                    elif isinstance(item, list) and len(item) == 4:
                        pred_boxes.append(item)
        except:
            pass
        
        # Parse ground truth
        gt_boxes = []
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except:
                pass
        
        if isinstance(ground_truth, list):
            for item in ground_truth:
                if isinstance(item, dict) and 'bbox_2d' in item:
                    gt_boxes.append(item['bbox_2d'])
                elif isinstance(item, list) and len(item) == 4:
                    gt_boxes.append(item)
        
        # Compute IoU matching
        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0
        
        matched = 0
        for gt in gt_boxes:
            for pred in pred_boxes:
                if compute_iou(pred, gt) >= 0.5:
                    matched += 1
                    break
        
        score = 1.0 if matched == len(gt_boxes) and gt_boxes else 0.0
        return {'score': score, 'matched': matched, 'total_gt': len(gt_boxes), 'total_pred': len(pred_boxes)}
    
    def evaluate_referring(self, response: str, ground_truth: str) -> Dict:
        """Evaluate referring task."""
        import regex as re
        
        def clean_text(text):
            if not text:
                return ""
            text = str(text).lower()
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = ' '.join(text.split())
            return text.strip()
        
        # Extract action from response
        pred = clean_text(response)
        gt = clean_text(ground_truth)
        
        # Check for action pattern
        action_match = re.search(r'action[:\s]+(.+?)(?:\n|$)', pred, re.IGNORECASE)
        if action_match:
            pred = clean_text(action_match.group(1))
        
        exact_match = pred == gt
        
        # Word overlap
        pred_words = set(pred.split())
        gt_words = set(gt.split())
        overlap = len(pred_words & gt_words) / len(gt_words) if gt_words else 0
        
        return {'score': 1.0 if exact_match else overlap, 'exact_match': exact_match, 'overlap': overlap}


async def run_evaluation(args):
    """Run evaluation on validation data."""
    evaluator = HOIAgentEvaluator(
        endpoint=args.endpoint,
        model_name=args.model,
        max_turns=args.max_turns,
        verbose=args.verbose
    )
    
    print("=" * 60)
    print("HOI Agent Evaluation")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Max turns: {args.max_turns}")
    print("=" * 60)
    
    # Load validation data
    df = pd.read_parquet(args.val_data)
    if args.max_samples:
        df = df.head(args.max_samples)
    
    print(f"\nEvaluating {len(df)} samples...")
    
    results = []
    metrics = defaultdict(list)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            # Get image path
            images_info = row.get('images', [])
            if not images_info:
                continue
            
            image_path = images_info[0].get('path', '') if isinstance(images_info[0], dict) else str(images_info[0])
            if not os.path.exists(image_path):
                continue
            
            # Get prompt and ground truth
            prompt = row.get('prompt', '')
            if isinstance(prompt, list):
                # Extract text from prompt messages
                prompt_text = ""
                for p in prompt:
                    if isinstance(p, dict) and 'content' in p:
                        prompt_text += str(p['content']) + "\n"
                prompt = prompt_text
            
            reward_model = row.get('reward_model', {})
            if isinstance(reward_model, str):
                reward_model = json.loads(reward_model)
            
            task_type = reward_model.get('task_type', 'grounding')
            ground_truth = reward_model.get('ground_truth', '')
            
            if args.verbose:
                print(f"\n[{idx}] {task_type}: {image_path}")
            
            # Run agent
            result = await evaluator.run_agent_loop(image_path, prompt, ground_truth)
            
            # Evaluate
            if task_type == 'grounding':
                eval_result = evaluator.evaluate_grounding(result['response'], ground_truth)
            else:
                eval_result = evaluator.evaluate_referring(result['response'], ground_truth)
            
            metrics[f'{task_type}_score'].append(eval_result['score'])
            metrics['overall_score'].append(eval_result['score'])
            metrics['num_tool_calls'].append(len(result['tool_calls']))
            
            results.append({
                'idx': idx,
                'task_type': task_type,
                'ground_truth': str(ground_truth),
                'prediction': result['response'],
                'score': eval_result['score'],
                'tool_calls': result['tool_calls'],
                'num_turns': result['num_turns']
            })
            
        except Exception as e:
            if args.verbose:
                print(f"Error: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for key, values in metrics.items():
        if values:
            print(f"{key}: {sum(values)/len(values):.4f} (n={len(values)})")
    print("=" * 60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    summary = {key: sum(values)/len(values) if values else 0 for key, values in metrics.items()}
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")


def build_referring_prompt(person_bbox: List[int], object_bbox: List[int], object_label: str = "object") -> str:
    """Build referring task prompt matching training format exactly."""
    return f"""Action Recognition Task: The first region {{"bbox_2d": {person_bbox}, "label": "person"}} contains a PERSON. The second region {{"bbox_2d": {object_bbox}, "label": "{object_label}"}} contains an OBJECT. Describe the action the person is performing with this object. Respond with only the action phrase (e.g., "riding bicycle", "sitting on bench").

Guidelines: Analyze the provided bounding boxes to determine what action the person is performing with the object. You may use zoom_in to examine interaction details. Output only the action phrase (e.g., "riding bicycle", "sitting on bench"). Use base verb form without articles."""


def build_grounding_prompt(action: str, object_label: str = "object") -> str:
    """Build grounding task prompt matching training format."""
    return f"""Human-Object Interaction Detection Task: Find all instances of "{action}" in this image.

For each interaction found, output the bounding boxes for:
1. The PERSON performing the action
2. The {object_label.upper()} involved in the action

Output format: List of {{"bbox_2d": [x1, y1, x2, y2], "label": "person/object"}} pairs.

Guidelines: Use detect_objects and zoom_in tools to locate people and objects. For each person-object pair performing "{action}", output their bounding boxes. Coordinates should be in image pixel units."""


def visualize_bboxes(image_path: str, predicted_boxes: List[Dict], gt_boxes: List[Dict] = None, 
                     save_path: str = None) -> Image.Image:
    """
    Draw bounding boxes on image for visualization.
    
    Args:
        image_path: Path to the image
        predicted_boxes: List of predicted boxes [{"bbox_2d": [x1,y1,x2,y2], "label": "..."}]
        gt_boxes: Optional ground truth boxes for comparison
        save_path: Optional path to save the visualization
        
    Returns:
        PIL Image with boxes drawn
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw ground truth boxes in green (dashed style simulated with multiple rectangles)
    if gt_boxes:
        for box_info in gt_boxes:
            bbox = box_info.get('bbox_2d', [])
            label = box_info.get('label', 'unknown')
            if len(bbox) == 4:
                # Scale from 1000-scale to image size if needed
                x1, y1, x2, y2 = bbox
                if max(bbox) <= 1000:  # Likely 1000-scale
                    x1 = int(x1 * width / 1000)
                    y1 = int(y1 * height / 1000)
                    x2 = int(x2 * width / 1000)
                    y2 = int(y2 * height / 1000)
                
                # Draw green box for GT
                color = (0, 200, 0)  # Green for GT
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1 + 5, y1 + 5), f"GT: {label}", fill=color, font=font)
    
    # Draw predicted boxes in red/blue
    colors = {'person': (255, 100, 100), 'object': (100, 100, 255)}  # Red for person, blue for object
    for i, box_info in enumerate(predicted_boxes):
        bbox = box_info.get('bbox_2d', [])
        label = box_info.get('label', 'unknown')
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Scale from 1000-scale to image size if needed
            if max(bbox) <= 1000:  # Likely 1000-scale
                x1 = int(x1 * width / 1000)
                y1 = int(y1 * height / 1000)
                x2 = int(x2 * width / 1000)
                y2 = int(y2 * height / 1000)
            
            color = colors.get(label.lower(), (255, 165, 0))  # Orange for others
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 5, y2 - 20), f"Pred: {label}", fill=color, font=font)
    
    if save_path:
        img.save(save_path)
        print(f"Visualization saved to: {save_path}")
    
    return img


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def extract_boxes_from_response(response: str) -> List[Dict]:
    """Extract bounding boxes from model response."""
    import re
    boxes = []
    
    # Pattern to match {"bbox_2d": [...], "label": "..."}
    pattern = r'\{\s*"bbox_2d"\s*:\s*\[([^\]]+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, response)
    
    for coords_str, label in matches:
        try:
            coords = [int(float(x.strip())) for x in coords_str.split(',')]
            if len(coords) == 4:
                boxes.append({"bbox_2d": coords, "label": label})
        except:
            continue
    
    # Also try to find bbox patterns like [x1, y1, x2, y2]
    if not boxes:
        bbox_pattern = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
        bbox_matches = re.findall(bbox_pattern, response)
        for i, (x1, y1, x2, y2) in enumerate(bbox_matches):
            label = "person" if i % 2 == 0 else "object"
            boxes.append({"bbox_2d": [int(x1), int(y1), int(x2), int(y2)], "label": label})
    
    return boxes


async def run_interactive(args):
    """Run interactive testing on a single image."""
    evaluator = HOIAgentEvaluator(
        endpoint=args.endpoint,
        model_name=args.model,
        max_turns=args.max_turns,
        verbose=True
    )
    
    print("=" * 60)
    print("HOI Agent Interactive Testing")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    
    task_type = "referring"  # Default
    gt_boxes = None
    
    # Check if this is a grounding task (has --action but no bounding boxes)
    if args.action and not (args.person_bbox and args.object_bbox):
        task_type = "grounding"
        object_label = args.object_label or "object"
        
        print(f"Mode: GROUNDING (find bounding boxes)")
        print(f"Action to ground: {args.action}")
        print(f"Object type: {object_label}")
        
        # Parse ground truth boxes if provided
        if args.ground_truth:
            try:
                gt_boxes = json.loads(args.ground_truth)
                print(f"Ground truth boxes: {len(gt_boxes)} boxes")
                for box in gt_boxes:
                    print(f"  - {box['label']}: {box['bbox_2d']}")
            except:
                print(f"Ground truth: {args.ground_truth}")
        
        prompt = build_grounding_prompt(args.action, object_label)
    
    # Check if using referring mode with bounding boxes
    elif args.person_bbox and args.object_bbox:
        task_type = "referring"
        person_bbox = json.loads(args.person_bbox)
        object_bbox = json.loads(args.object_bbox)
        object_label = args.object_label or "object"
        
        print(f"Mode: REFERRING (with bounding boxes)")
        print(f"Person bbox (1000 scale): {person_bbox}")
        print(f"Object bbox (1000 scale): {object_bbox}")
        if args.ground_truth:
            print(f"Ground truth: {args.ground_truth}")
        
        prompt = build_referring_prompt(person_bbox, object_bbox, object_label)
    else:
        print(f"Mode: GENERAL (no bounding boxes)")
        # Default prompt for testing
        prompt = args.prompt or """Analyze this image and identify any human-object interactions.

If this is a GROUNDING task: Find the person and object involved in the interaction and output their bounding boxes.
If this is a REFERRING task: Describe the action being performed.

Use the available tools (zoom_in, zoom_out, detect_objects) to help analyze the image."""
    
    print("=" * 60)
    print(f"\nPrompt:\n{prompt}\n")
    print("-" * 60)
    
    result = await evaluator.run_agent_loop(args.image_path, prompt)
    
    print("\n" + "=" * 60)
    print("Final Response:")
    print("=" * 60)
    print(result['response'])
    print("\n" + "-" * 60)
    print(f"Tool calls made: {len(result['tool_calls'])}")
    for tc in result['tool_calls']:
        print(f"  - {tc['name']}: {tc['arguments']}")
    print(f"Total turns: {result['num_turns']}")
    
    # For grounding tasks, extract and visualize predicted boxes
    if task_type == "grounding":
        print("\n" + "-" * 60)
        print("GROUNDING EVALUATION:")
        
        # Extract predicted boxes from response
        predicted_boxes = extract_boxes_from_response(result['response'])
        print(f"  Predicted boxes: {len(predicted_boxes)}")
        for box in predicted_boxes:
            print(f"    - {box['label']}: {box['bbox_2d']}")
        
        # Create visualization
        if predicted_boxes or gt_boxes:
            viz_path = args.output_viz or f"/tmp/hoi_grounding_viz_{datetime.now().strftime('%H%M%S')}.jpg"
            viz_img = visualize_bboxes(
                args.image_path, 
                predicted_boxes, 
                gt_boxes,
                save_path=viz_path
            )
            print(f"\n  Visualization saved: {viz_path}")
            print("  Legend: GREEN = Ground Truth, RED/BLUE = Predicted (person/object)")
        
        # Calculate IoU if ground truth is available
        if gt_boxes and predicted_boxes:
            print("\n  Box Matching Analysis:")
            for gt_box in gt_boxes:
                gt_bbox = gt_box['bbox_2d']
                gt_label = gt_box['label']
                best_iou = 0
                best_pred = None
                for pred_box in predicted_boxes:
                    if pred_box['label'].lower() == gt_label.lower():
                        iou = compute_iou(gt_bbox, pred_box['bbox_2d'])
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = pred_box
                if best_pred:
                    print(f"    GT {gt_label} {gt_bbox} -> Pred {best_pred['bbox_2d']}, IoU: {best_iou:.3f}")
                else:
                    print(f"    GT {gt_label} {gt_bbox} -> No matching prediction")
        
        print("=" * 60)
        return
    
    # Check against ground truth if provided (for referring tasks)
    if args.ground_truth:
        response = result['response']
        gt = args.ground_truth.lower().strip()
        
        # Extract action from response using multiple strategies
        import re
        predicted = None
        
        # Strategy 1: Look for "Final Answer:" or "**Final Answer:**" or "final answer."
        final_match = re.search(r'\*?\*?final\s*answer\*?\*?[:\.\s]+\s*([^\n]+)', response, re.IGNORECASE)
        if final_match:
            predicted = final_match.group(1).strip().lower()
            # Clean up markdown/formatting from the extracted answer
            predicted = re.sub(r'\*+', '', predicted).strip()
        
        # Strategy 2: Look for "action (phrase) is:" or "ACTION:" pattern
        if not predicted:
            action_match = re.search(r'(?:action\s*(?:phrase\s*)?(?:is)?|action)[:\.\s]+\s*([^\n]+)', response, re.IGNORECASE)
            if action_match:
                # Skip if we captured "phrase is" or similar meta text
                captured = action_match.group(1).strip().lower()
                if captured and captured not in ['phrase is', 'is', 'phrase']:
                    predicted = captured
        
        # Strategy 3: Look for the ground truth pattern directly in response
        if not predicted:
            # Check if GT appears in response
            if gt in response.lower():
                predicted = gt
        
        # Strategy 4: Look for standalone lines that look like actions (2-4 words)
        # Check from the end of the response first (final answer is usually last)
        if not predicted:
            lines = response.split('\n')
            for line in reversed(lines):
                line_clean = line.strip().lower()
                # Skip empty lines, lines with markdown, or too long
                if not line_clean or line_clean.startswith('#') or '*' in line_clean or ':' in line_clean or len(line_clean) > 40:
                    continue
                # Check if it's a short action-like phrase
                words = line_clean.split()
                if 2 <= len(words) <= 4:
                    # First word should be a verb (ends in -ing typically for actions)
                    if words[0].endswith('ing') or words[0] in ['hold', 'ride', 'sit', 'eat', 'use', 'play']:
                        predicted = line_clean.rstrip('.!?,;:')
                        break
        
        # Strategy 5: Look for verb + (optional preposition) + object patterns
        # Use finditer and take the last match (usually the final answer)
        if not predicted:
            verbs = r'(?:sitting|holding|riding|carrying|eating|drinking|reading|watching|using|playing|walking|running|standing|lying|pushing|pulling|throwing|catching|kicking|hitting|cutting|washing|cleaning|cooking|driving|flying|swimming|climbing|jumping|dancing|singing|writing|drawing|painting|typing|calling|talking|listening|looking|waiting|sleeping|waking|opening|closing|turning|moving|lifting|dropping|picking|putting|taking|giving|receiving|sending|buying|selling|making|building|fixing|breaking|tearing|folding|wrapping|packing|feeding|petting|leading|training|brushing|grooming)'
            preps = r'(?:\s+(?:on|with|at|in|to|from|into|onto|off|up|down|over|under|through|across|around|behind|beside|between|inside|outside|above|below|near))?'
            pattern = rf'\b{verbs}{preps}\s+[a-z]+\b'
            all_matches = list(re.finditer(pattern, response.lower()))
            if all_matches:
                # Take the last match (final answer is usually at the end)
                predicted = all_matches[-1].group(0).strip()
        
        # Strategy 6: Last resort - take last short meaningful line
        if not predicted:
            lines = [l.strip() for l in response.split('\n') if l.strip() and len(l.strip()) < 50]
            if lines:
                predicted = lines[-1].lower().rstrip('.!?,;:')
            else:
                predicted = response[:50].lower()
        
        # Clean up predicted
        predicted = predicted.strip().rstrip('.!?,;:')
        # Remove common prefixes
        for prefix in ['the action is ', 'answer: ', 'action: ', 'the person is ']:
            if predicted.startswith(prefix):
                predicted = predicted[len(prefix):]
        
        print("\n" + "-" * 60)
        print("EVALUATION:")
        print(f"  Ground truth: {args.ground_truth}")
        print(f"  Extracted:    {predicted}")
        
        # Check for match
        exact_match = predicted == gt
        contains_match = gt in response.lower()
        partial_match = all(word in predicted for word in gt.split())
        
        if exact_match:
            print(f"  Match:        ✓ EXACT MATCH")
        elif contains_match:
            print(f"  Match:        ✓ FOUND IN RESPONSE")
        elif partial_match:
            print(f"  Match:        ~ PARTIAL MATCH")
        else:
            print(f"  Match:        ✗ NO MATCH")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="HOI Agent Evaluation with Tool Calling")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1",
                        help="vLLM server endpoint")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name served by vLLM")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Maximum agent turns (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    # Evaluation mode
    parser.add_argument("--val-data", type=str, default="data/hoi/train_data/val.parquet",
                        help="Validation data path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate")
    parser.add_argument("--output-dir", type=str, default="results/hoi_agent_eval",
                        help="Output directory")
    
    # Interactive mode
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive testing on single image")
    parser.add_argument("--image-path", type=str,
                        help="Image path for interactive testing")
    parser.add_argument("--prompt", type=str,
                        help="Custom prompt for interactive testing")
    
    # Referring task with bounding boxes (matches training format)
    parser.add_argument("--person-bbox", type=str,
                        help="Person bounding box [x1,y1,x2,y2] in 1000-scale (e.g., '[500,718,561,819]')")
    parser.add_argument("--object-bbox", type=str,
                        help="Object bounding box [x1,y1,x2,y2] in 1000-scale (e.g., '[231,809,588,971]')")
    parser.add_argument("--object-label", type=str, default="object",
                        help="Object label (e.g., 'bench', 'horse')")
    parser.add_argument("--ground-truth", type=str,
                        help="Ground truth action for evaluation (e.g., 'sitting on bench') or JSON boxes for grounding")
    
    # Grounding task arguments
    parser.add_argument("--action", type=str,
                        help="Action to ground (e.g., 'sitting on', 'holding'). Use with --object-label for grounding tasks.")
    parser.add_argument("--output-viz", type=str,
                        help="Path to save visualization image (default: /tmp/hoi_grounding_viz_HHMMSS.jpg)")
    
    args = parser.parse_args()
    
    if args.interactive:
        if not args.image_path:
            print("Error: --image-path required for interactive mode")
            sys.exit(1)
        asyncio.run(run_interactive(args))
    else:
        asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()

