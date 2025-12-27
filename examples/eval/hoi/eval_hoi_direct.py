#!/usr/bin/env python3
"""
HOI Detection Direct Evaluation

Directly loads the trained model and runs inference matching the training format.
Does NOT require vLLM server - loads model directly with transformers.

Usage:
    python examples/eval/hoi/eval_hoi_direct.py \
        --model-path checkpoints/.../global_step_100/actor/huggingface \
        --image-path data/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
        --task referring \
        --verbose
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# System prompt matching training format exactly
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


def load_model(model_path: str, device: str = "cuda"):
    """Load the trained model and processor."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    print(f"Loading model from: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Use AutoModelForVision2Seq which handles both Qwen2-VL and Qwen3-VL
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, processor


def build_referring_prompt(person_bbox: List[int], object_bbox: List[int], object_label: str = "object"):
    """Build referring task prompt matching training format."""
    return f"""Action Recognition Task: The first region {{"bbox_2d": {person_bbox}, "label": "person"}} contains a PERSON. The second region {{"bbox_2d": {object_bbox}, "label": "{object_label}"}} contains an OBJECT. Describe the action the person is performing with this object. Respond with only the action phrase (e.g., "riding bicycle", "sitting on bench").

Guidelines: Analyze the provided bounding boxes to determine what action the person is performing with the object. You may use zoom_in to examine interaction details. Output only the action phrase (e.g., "riding bicycle", "sitting on bench"). Use base verb form without articles."""


def build_grounding_prompt(action: str, object_label: str = "object"):
    """Build grounding task prompt matching training format."""
    return f"""Object Grounding Task: Find the person and {object_label} that are involved in the action "{action}".

Guidelines: Detect objects using detect_objects tool if needed. Output bounding boxes in format: [{{"bbox_2d": [x1, y1, x2, y2], "label": "person"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "{object_label}"}}]"""


def parse_tool_calls(response: str) -> List[Dict]:
    """Parse tool calls from model response."""
    tool_calls = []
    pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
    matches = re.findall(pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            pass
    
    return tool_calls


def execute_tool(tool_name: str, arguments: Dict, current_image: Image.Image, original_image: Image.Image) -> tuple:
    """Execute a tool and return (result_text, new_current_image)."""
    if tool_name == 'zoom_in':
        bbox = arguments.get('bbox_2d', [])
        if len(bbox) == 4:
            # Convert from 1000x1000 to actual pixels
            w, h = current_image.size
            x1 = int(bbox[0] / 1000 * w)
            y1 = int(bbox[1] / 1000 * h)
            x2 = int(bbox[2] / 1000 * w)
            y2 = int(bbox[3] / 1000 * h)
            cropped = current_image.crop((x1, y1, x2, y2))
            return f"Zoomed into region {bbox}. Now viewing a {cropped.size[0]}x{cropped.size[1]} cropped area.", cropped
        return "Invalid bbox format.", current_image
    
    elif tool_name == 'zoom_out':
        return f"Zoomed out to full image view ({original_image.size[0]}x{original_image.size[1]}).", original_image.copy()
    
    elif tool_name == 'detect_objects':
        class_names = arguments.get('class_names', 'person . object')
        try:
            from verl_tool.servers.tools.hoi_detector import detect_objects_with_grounding_dino
            detections = detect_objects_with_grounding_dino(current_image, class_names)
            if detections:
                result = "Detected objects:\n"
                for det in detections:
                    result += f"  - {det['label']}: bbox={det['bbox']}, confidence={det['confidence']:.2f}\n"
                return result, current_image
            return "No objects detected.", current_image
        except Exception as e:
            return f"Detection failed: {str(e)}", current_image
    
    return f"Unknown tool: {tool_name}", current_image


def run_inference(model, processor, image: Image.Image, user_prompt: str, max_turns: int = 5, verbose: bool = False):
    """Run multi-turn inference with tool calling."""
    
    current_image = image.copy()
    original_image = image.copy()
    conversation = []
    tool_calls_made = []
    
    # Build initial messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    
    for turn in range(max_turns):
        if verbose:
            print(f"\n  Turn {turn + 1}/{max_turns}")
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=[current_image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        
        # Decode only the new tokens
        input_len = inputs['input_ids'].shape[1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        if verbose:
            print(f"    Response: {response[:200]}...")
        
        # Check for tool calls
        tool_calls = parse_tool_calls(response)
        
        if tool_calls:
            # Execute tools
            for tc in tool_calls:
                tool_name = tc.get('name', '')
                arguments = tc.get('arguments', {})
                
                if verbose:
                    print(f"    [TOOL] {tool_name}({arguments})")
                
                result, current_image = execute_tool(tool_name, arguments, current_image, original_image)
                tool_calls_made.append({'name': tool_name, 'arguments': arguments, 'result': result})
                
                if verbose:
                    print(f"    [RESULT] {result[:100]}...")
            
            # Add assistant response and tool result to messages
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool result: {result}"})
            
        else:
            # No tool calls - this is the final response
            conversation.append({'role': 'assistant', 'content': response})
            return {
                'response': response,
                'tool_calls': tool_calls_made,
                'num_turns': turn + 1
            }
    
    # Max turns reached
    return {
        'response': response if 'response' in dir() else "",
        'tool_calls': tool_calls_made,
        'num_turns': max_turns
    }


def main():
    parser = argparse.ArgumentParser(description="HOI Direct Evaluation")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--image-path", type=str, required=True, help="Path to test image")
    parser.add_argument("--task", type=str, choices=['referring', 'grounding'], default='referring', help="Task type")
    parser.add_argument("--person-bbox", type=str, default="[100, 100, 500, 800]", help="Person bbox for referring task")
    parser.add_argument("--object-bbox", type=str, default="[300, 300, 700, 900]", help="Object bbox for referring task")
    parser.add_argument("--action", type=str, default="riding bicycle", help="Action for grounding task")
    parser.add_argument("--object-label", type=str, default="object", help="Object label")
    parser.add_argument("--max-turns", type=int, default=5, help="Max agent turns")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Compare with base model
    parser.add_argument("--compare-base", action="store_true", help="Also run with base model for comparison")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Base model for comparison")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HOI Direct Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print(f"Task: {args.task}")
    print("=" * 60)
    
    # Load image
    image = Image.open(args.image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Build prompt based on task
    if args.task == 'referring':
        person_bbox = json.loads(args.person_bbox)
        object_bbox = json.loads(args.object_bbox)
        prompt = build_referring_prompt(person_bbox, object_bbox, args.object_label)
    else:
        prompt = build_grounding_prompt(args.action, args.object_label)
    
    print(f"\nPrompt:\n{prompt[:500]}...")
    print("-" * 60)
    
    # Load trained model
    model, processor = load_model(args.model_path, args.device)
    
    # Run inference
    print("\n[TRAINED MODEL]")
    result = run_inference(model, processor, image, prompt, args.max_turns, args.verbose)
    
    print("\n" + "=" * 60)
    print("TRAINED MODEL RESULT:")
    print("=" * 60)
    print(f"Response: {result['response']}")
    print(f"Tool calls: {len(result['tool_calls'])}")
    for tc in result['tool_calls']:
        print(f"  - {tc['name']}: {tc['arguments']}")
    print(f"Turns: {result['num_turns']}")
    
    # Compare with base model if requested
    if args.compare_base:
        print("\n" + "=" * 60)
        print("[BASE MODEL]")
        print("=" * 60)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        
        base_model, base_processor = load_model(args.base_model, args.device)
        base_result = run_inference(base_model, base_processor, image, prompt, args.max_turns, args.verbose)
        
        print("\nBASE MODEL RESULT:")
        print("=" * 60)
        print(f"Response: {base_result['response']}")
        print(f"Tool calls: {len(base_result['tool_calls'])}")
        print(f"Turns: {base_result['num_turns']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

