#!/usr/bin/env python3
"""
Test script for HOI model with tool use (zoom_in) capability.
Tests the multi-turn conversation where model calls zoom_in tool.
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


def test_tool_use(model, processor, sample, images_dir, device="cuda:0"):
    """Test a sample that requires tool use."""
    
    # Get images
    image_paths = sample.get("images", [])
    if len(image_paths) < 2:
        print("Sample doesn't have 2 images for tool use test")
        return None
    
    image1_path = images_dir / image_paths[0]
    image2_path = images_dir / image_paths[1]
    
    if not image1_path.exists() or not image2_path.exists():
        print(f"Images not found: {image1_path}, {image2_path}")
        return None
    
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    
    # Parse expected responses from dataset
    messages_data = sample.get("messages", [])
    user_question = None
    expected_tool_call = None
    expected_final_answer = None
    
    for msg in messages_data:
        content = msg.get("content", "")
        if msg.get("role") == "user" and "Question:" in content:
            user_question = content.replace("<image> ", "").replace("<image>", "")
        elif msg.get("role") == "assistant":
            if "<tool_call>" in content:
                expected_tool_call = content
            elif "<answer>" in content:
                expected_final_answer = content
    
    if not user_question:
        print("Could not parse user question")
        return None
    
    print("\n" + "=" * 60)
    print("TURN 1: Initial image + question")
    print("=" * 60)
    
    # Turn 1: Send initial image and question
    messages_turn1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "text", "text": user_question}
            ]
        }
    ]
    
    response1 = generate_response(model, processor, messages_turn1, [image1], device)
    
    print(f"\nUser Question:")
    print("-" * 40)
    # Extract just the question part
    q_match = re.search(r'Question:(.+?)(?:The person|Respond)', user_question, re.DOTALL)
    if q_match:
        print(q_match.group(1).strip())
    print("-" * 40)
    
    print(f"\nModel Response (Turn 1):")
    print("-" * 40)
    print(response1)
    print("-" * 40)
    
    # Check if model called tool
    tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response1, re.DOTALL)
    
    if tool_call_match:
        print("\n✅ Model called zoom_in tool!")
        tool_call_json = tool_call_match.group(1)
        try:
            tool_call = json.loads(tool_call_json)
            print(f"   Tool: {tool_call.get('name')}")
            print(f"   Args: {tool_call.get('arguments')}")
        except json.JSONDecodeError:
            print(f"   Raw: {tool_call_json}")
        
        print("\n" + "=" * 60)
        print("TURN 2: Zoomed image (simulated tool response)")
        print("=" * 60)
        
        # Turn 2: Send zoomed image as tool response
        follow_up_prompt = "Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
        
        messages_turn2 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1},
                    {"type": "text", "text": user_question}
                ]
            },
            {"role": "assistant", "content": response1},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2},
                    {"type": "text", "text": follow_up_prompt}
                ]
            }
        ]
        
        response2 = generate_response(model, processor, messages_turn2, [image1, image2], device)
        
        print(f"\nModel Response (Turn 2 - After seeing zoomed image):")
        print("-" * 40)
        print(response2)
        print("-" * 40)
        
        # Extract final answer
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response2)
        if answer_match:
            predicted_action = answer_match.group(1)
            print(f"\n🎯 Predicted Action: {predicted_action}")
        
        if expected_final_answer:
            exp_match = re.search(r'<answer>\s*(.*?)\s*</answer>', expected_final_answer)
            if exp_match:
                print(f"📋 Expected Action: {exp_match.group(1)}")
    else:
        print("\n⚠️ Model did NOT call tool - gave direct answer")
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response1)
        if answer_match:
            print(f"🎯 Predicted Action: {answer_match.group(1)}")
    
    return response1


def main():
    print("=" * 60)
    print("HOI Chain-of-Focus Tool Use Test")
    print("=" * 60)
    
    # Load model
    model, processor = load_model_with_lora()
    
    # Load dataset
    images_dir = Path("/workspace/verl-tool/data/hoi_cof_sft/images")
    dataset_path = "/workspace/verl-tool/data/hoi_cof_sft/hoi_cof_sft_data.json"
    
    print("\nLoading dataset...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Find samples with tool use (2 images)
    tool_samples = []
    for i, sample in enumerate(dataset):
        if len(sample.get("images", [])) == 2:
            messages = sample.get("messages", [])
            has_tool = any("<tool_call>" in msg.get("content", "") for msg in messages)
            if has_tool:
                tool_samples.append((i, sample))
        if len(tool_samples) >= 3:
            break
    
    print(f"\nFound {len(tool_samples)} samples with tool use")
    
    # Test samples
    for idx, (sample_idx, sample) in enumerate(tool_samples[:2]):  # Test 2 samples
        print(f"\n\n{'#' * 60}")
        print(f"# TEST {idx + 1} (Dataset sample {sample_idx})")
        print(f"{'#' * 60}")
        
        test_tool_use(model, processor, sample, images_dir)
    
    print("\n\n" + "=" * 60)
    print("Tool Use Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
