#!/usr/bin/env python3
"""
Merge LoRA weights into base model for vLLM compatibility.

The GRPO training saves models with PEFT structure:
  - base_model.model.* prefix
  - Separate lora_A/lora_B weights

vLLM needs a standard HuggingFace model without PEFT wrapper.
This script merges LoRA weights and saves a vLLM-compatible checkpoint.

Usage:
    python scripts/merge_lora_for_vllm.py \
        --base-model Qwen/Qwen3-VL-8B-Instruct \
        --lora-path checkpoints/actor/lora_adapter \
        --output-dir checkpoints/actor/merged_for_vllm
"""

import os
import sys
import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights for vLLM")
    parser.add_argument("--base-model", type=str, required=True,
                        help="Base model name or path (e.g., Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for merged model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (default: bfloat16)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Merging LoRA weights for vLLM compatibility")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Dtype: {args.dtype}")
    print("=" * 60)
    
    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Load base model
    print("\n[1/4] Loading base model...")
    from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration
    
    # Try to load as Qwen3VL or Qwen2VL
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        print("  Loaded as Qwen3VLForConditionalGeneration")
    except:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        print("  Loaded as Qwen2VLForConditionalGeneration")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Load LoRA adapter
    print("\n[2/4] Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.lora_path)
    print(f"  LoRA adapter loaded from {args.lora_path}")
    
    # Merge LoRA weights
    print("\n[3/4] Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    print("  LoRA weights merged successfully")
    
    # Save merged model
    print(f"\n[4/4] Saving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"\nMerged model saved to: {args.output_dir}")
    print("\nYou can now start vLLM server with:")
    print(f"  CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/eval/hoi/start_vllm_server.sh \\")
    print(f"      {args.output_dir} 8000 hoi-grpo 4")


if __name__ == "__main__":
    main()
