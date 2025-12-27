#!/usr/bin/env python3
"""
HOI Detection Evaluation Script

Evaluates a trained model on the HOI validation dataset.

Usage:
    python examples/eval/hoi/eval_hoi.py \
        --model_path checkpoints/hoi_reward/.../global_step_100/actor/huggingface \
        --val_data data/hoi/train_data/val.parquet \
        --output_dir results/hoi_eval
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_model_and_processor(model_path: str, device: str = "cuda"):
    """Load the trained model and processor."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    print(f"Loading model from: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, processor


def load_image(image_info: dict) -> Image.Image:
    """Load image from path info."""
    if isinstance(image_info, dict):
        image_path = image_info.get('path', image_info.get('image_path', ''))
    else:
        image_path = str(image_info)
    
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    else:
        raise FileNotFoundError(f"Image not found: {image_path}")


def extract_response(full_output: str, prompt: str) -> str:
    """Extract just the model response from the full output."""
    # Remove the prompt from the beginning
    if prompt in full_output:
        response = full_output.split(prompt)[-1]
    else:
        response = full_output
    return response.strip()


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def extract_boxes_from_response(response: str):
    """Extract bounding boxes from model response."""
    import regex as re
    
    boxes = []
    try:
        # Try JSON parsing
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
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
    except:
        pass
    
    # Fallback: regex
    if not boxes:
        pattern = r'[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]'
        matches = re.findall(pattern, response)
        for m in matches:
            boxes.append([float(m[0]), float(m[1]), float(m[2]), float(m[3])])
    
    return boxes


def evaluate_grounding(pred_response: str, ground_truth: list, iou_threshold: float = 0.5):
    """Evaluate grounding task."""
    pred_boxes = extract_boxes_from_response(pred_response)
    
    # Parse ground truth boxes
    gt_boxes = []
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except:
            ground_truth = []
    
    if isinstance(ground_truth, list):
        for item in ground_truth:
            if isinstance(item, dict) and 'bbox_2d' in item:
                gt_boxes.append(item['bbox_2d'])
            elif isinstance(item, list) and len(item) == 4:
                gt_boxes.append(item)
    
    if not pred_boxes or not gt_boxes:
        return 0.0, {"matched": 0, "total_gt": len(gt_boxes), "total_pred": len(pred_boxes)}
    
    # Match predictions to ground truth
    matched = 0
    used_preds = set()
    
    for gt_box in gt_boxes:
        best_iou = 0
        best_idx = -1
        for idx, pred_box in enumerate(pred_boxes):
            if idx in used_preds:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        
        if best_iou >= iou_threshold and best_idx >= 0:
            matched += 1
            used_preds.add(best_idx)
    
    score = 1.0 if matched == len(gt_boxes) else 0.0
    return score, {"matched": matched, "total_gt": len(gt_boxes), "total_pred": len(pred_boxes)}


def evaluate_referring(pred_response: str, ground_truth: str):
    """Evaluate referring task with word overlap."""
    import regex as re
    
    def clean_text(text):
        if text is None:
            return ""
        text = str(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\1', text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = ' '.join(text.split())
        return text.strip().lower()
    
    pred_clean = clean_text(pred_response)
    gt_clean = clean_text(ground_truth)
    
    # Exact match
    if pred_clean == gt_clean:
        return 1.0, {"exact_match": True}
    
    # Word overlap
    pred_words = set(pred_clean.split())
    gt_words = set(gt_clean.split())
    if not gt_words:
        return 0.0, {"exact_match": False, "overlap": 0.0}
    
    overlap = len(pred_words & gt_words) / len(gt_words)
    return overlap, {"exact_match": False, "overlap": overlap}


def run_inference(model, processor, prompt: str, images: list, max_new_tokens: int = 512):
    """Run inference on a single sample."""
    # Build messages format
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process with images
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    # Decode
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Evaluate HOI detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--val_data", type=str, default="data/hoi/train_data/val.parquet", help="Validation data path")
    parser.add_argument("--output_dir", type=str, default="results/hoi_eval", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, processor = load_model_and_processor(args.model_path, args.device)
    
    # Load validation data
    print(f"Loading validation data from: {args.val_data}")
    df = pd.read_parquet(args.val_data)
    
    if args.max_samples:
        df = df.head(args.max_samples)
    
    print(f"Evaluating {len(df)} samples...")
    
    # Evaluate
    results = []
    metrics = defaultdict(list)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            # Load image
            images_info = row.get('images', [])
            images = []
            for img_info in images_info:
                try:
                    img = load_image(img_info)
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load image: {e}")
            
            # Get prompt
            prompt = row.get('prompt', '')
            
            # Run inference
            response = run_inference(model, processor, prompt, images, args.max_new_tokens)
            
            # Get task type and ground truth
            reward_model = row.get('reward_model', {})
            if isinstance(reward_model, str):
                reward_model = json.loads(reward_model)
            
            task_type = reward_model.get('task_type', 'grounding')
            ground_truth = reward_model.get('ground_truth', '')
            
            # Evaluate
            if task_type == 'grounding':
                score, details = evaluate_grounding(response, ground_truth)
                metrics['grounding_score'].append(score)
                metrics['grounding_matched'].append(details['matched'])
                metrics['grounding_total_gt'].append(details['total_gt'])
            else:
                score, details = evaluate_referring(response, ground_truth)
                metrics['referring_score'].append(score)
                if details.get('exact_match'):
                    metrics['referring_exact_match'].append(1.0)
                else:
                    metrics['referring_exact_match'].append(0.0)
            
            metrics['overall_score'].append(score)
            
            # Store result
            results.append({
                'idx': idx,
                'task_type': task_type,
                'ground_truth': str(ground_truth),
                'prediction': response,
                'score': score,
                'details': details
            })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute aggregate metrics
    summary = {
        'model_path': args.model_path,
        'val_data': args.val_data,
        'num_samples': len(results),
        'timestamp': datetime.now().isoformat(),
    }
    
    for key, values in metrics.items():
        if values:
            summary[f'{key}_mean'] = sum(values) / len(values)
            summary[f'{key}_count'] = len(values)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)
    
    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

