#!/usr/bin/env python3
"""
Test the HOI model on grounding tasks.
Given an action description, can the model locate the person and object?
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/verl-tool/LlamaFactory/src")

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel


def load_model_with_lora(
    base_model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    lora_path: str = "/workspace/verl-tool/LlamaFactory/saves/qwen3-vl-8b/lora/hoi_cof_sft",
    device: str = "cuda:0"
):
    """Load base model with LoRA adapters."""
    print(f"Loading base model: {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    print(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    print("Model loaded successfully!")
    return model, processor


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


def generate_response(model, processor, messages, images, device="cuda:0"):
    """Generate response from the model."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


def normalize_box(box, width, height):
    """Convert absolute box to 1000x1000 normalized format."""
    x1, y1, x2, y2 = box
    return [
        int(x1 * 1000 / width),
        int(y1 * 1000 / height),
        int(x2 * 1000 / width),
        int(y2 * 1000 / height)
    ]


def test_grounding(model, processor, sample, images_dir, device="cuda:0"):
    """Test grounding: given an action, locate person and object."""
    
    file_name = sample["file_name"]
    image_path = images_dir / file_name
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None
    
    image = Image.open(image_path).convert("RGB")
    width, height = sample["width"], sample["height"]
    action = sample["action"]
    obj_category = sample["object_category"]
    boxes = sample["boxes"]
    
    # Normalize ground truth boxes
    gt_boxes_norm = [normalize_box(b, width, height) for b in boxes]
    
    # Create grounding query
    query = f"""Question: Locate every person who is {action} {obj_category} and the {obj_category} they interact with.
For each person-object pair, output bbox coordinates in JSON format like: {{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}. Coordinates should be in 1000x1000 normalized format.
Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]
        }
    ]
    
    print(f"\n{'=' * 60}")
    print(f"Image: {file_name}")
    print(f"Query: Locate person {action} {obj_category}")
    print(f"Ground Truth Boxes (normalized): {gt_boxes_norm[:4]}")  # Show first 4
    print(f"{'=' * 60}")
    
    response = generate_response(model, processor, messages, [image], device)
    
    print(f"\nModel Response:")
    print("-" * 40)
    print(response)
    print("-" * 40)
    
    # Check if model used tools or gave direct answer
    if "<tool_call>" in response:
        print("✅ Model decided to zoom in for better detection")
    elif "<answer>" in response:
        print("✅ Model gave direct answer")
        # Try to extract bboxes from answer
        bbox_pattern = r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        pred_boxes = re.findall(bbox_pattern, response)
        if pred_boxes:
            print(f"📍 Predicted boxes: {[[int(x) for x in b] for b in pred_boxes]}")
    
    return response


def main():
    print("=" * 60)
    print("HOI Grounding Test")
    print("=" * 60)
    
    # Load model
    model, processor = load_model_with_lora()
    
    # Load grounding dataset
    grounding_path = "/workspace/verl-tool/data/benchmarks_simplified/hico_ground_test_simplified.json"
    images_dir = Path("/workspace/verl-tool/data/hico_20160224_det/images/test2015")
    
    print(f"\nLoading grounding dataset from: {grounding_path}")
    with open(grounding_path, 'r') as f:
        grounding_data = json.load(f)
    
    print(f"Total samples: {len(grounding_data)}")
    
    # Test on a few diverse samples
    test_indices = [0, 1, 10, 50]  # Different samples
    
    for idx in test_indices:
        if idx < len(grounding_data):
            sample = grounding_data[idx]
            print(f"\n\n{'#' * 60}")
            print(f"# TEST {idx + 1} (Sample index {idx})")
            print(f"{'#' * 60}")
            test_grounding(model, processor, sample, images_dir)
    
    print("\n\n" + "=" * 60)
    print("Grounding Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
