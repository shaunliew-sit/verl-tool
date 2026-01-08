#!/usr/bin/env python3
"""
Test script for the HOI Chain-of-Focus SFT trained model.
Tests the Qwen3-VL-8B model with LoRA adapters for Human-Object Interaction detection.
"""

import json
import os
import sys
from pathlib import Path

# Add LlamaFactory to path
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
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model using Auto class for Qwen3-VL
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    print("Model loaded successfully!")
    return model, processor


def test_hoi_detection(
    model, 
    processor, 
    image_path: str,
    person_bbox: list,
    object_bbox: list,
    device: str = "cuda:0"
):
    """Test HOI detection on a single image."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Create the prompt (matching training format)
    system_prompt = """You are a helpful assistant.

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

    user_prompt = f"""Question: What action is the person performing with the object?
The person is located at {person_bbox} and the object is at {object_bbox}.
Respond with ONLY the action phrase in format: "{{verb}} {{object}}" (e.g., "riding bicycle", "holding cup"). Use base verb form, no articles.
Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."""

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode response
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    return response


def main():
    import re
    
    print("=" * 60)
    print("HOI Chain-of-Focus Model Test")
    print("=" * 60)
    
    # Load model
    model, processor = load_model_with_lora()
    
    # Find test images
    test_images_dir = Path("/workspace/verl-tool/data/hoi_cof_sft/images")
    
    # Load a sample from the dataset to get proper bbox coordinates
    dataset_path = "/workspace/verl-tool/data/hoi_cof_sft/hoi_cof_sft_data.json"
    
    print("\nLoading test samples from dataset...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Test on first 3 samples
    num_tests = min(3, len(dataset))
    
    print(f"\nTesting on {num_tests} samples...")
    print("=" * 60)
    
    for i in range(num_tests):
        sample = dataset[i]
        
        # Get image path (new format uses "messages" and "images")
        image_paths = sample.get("images", [])
        if not image_paths:
            print(f"Sample {i}: No images found")
            continue
            
        image_path = test_images_dir / image_paths[0]
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
        
        # Extract info from messages format
        messages = sample.get("messages", [])
        user_msg = None
        expected_output = None
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and "Question:" in content:
                user_msg = content
            elif role == "assistant" and "<answer>" in content:
                expected_output = content
        
        if not user_msg:
            print(f"Sample {i}: No user message found")
            continue
        
        # Parse bboxes from user message
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        bboxes = re.findall(bbox_pattern, user_msg)
        
        if len(bboxes) >= 2:
            person_bbox = [int(x) for x in bboxes[0]]
            object_bbox = [int(x) for x in bboxes[1]]
        else:
            print(f"Sample {i}: Could not parse bboxes")
            continue
        
        print(f"\n[Test {i+1}]")
        print(f"Image: {image_path.name}")
        print(f"Person bbox: {person_bbox}")
        print(f"Object bbox: {object_bbox}")
        
        # Run inference
        response = test_hoi_detection(
            model, processor, 
            str(image_path),
            person_bbox, object_bbox
        )
        
        print(f"\nModel Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        if expected_output:
            # Extract answer from expected output
            answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', expected_output)
            if answer_match:
                print(f"Expected: {answer_match.group(1)}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
