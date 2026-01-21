#!/usr/bin/env python3
"""
Recalculate BERTScore using microsoft/deberta-v2-xxlarge-mnli model.

This script reads the per_sample_results.json from a previous evaluation run
and recalculates BERTScore with the specified model, updating the metrics.json.
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict


def clean_action_text(text: str) -> str:
    """Clean action text for comparison."""
    if not text:
        return ""
    
    # Remove <answer> tags if present
    text = re.sub(r'</?answer\s*>', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.lower().strip()


def compute_bertscore_deberta(
    predictions: List[Dict], 
    bertscore_gpu: int = 0,
    batch_size: int = 32
) -> Dict:
    """Compute BERTScore using microsoft/deberta-v2-xxlarge-mnli model."""
    
    pred_texts = []
    gt_texts = []
    
    for p in predictions:
        pred_texts.append(p.get('prediction', ''))
        gt_texts.append(p.get('ground_truth', ''))
    
    metrics = {}
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(bertscore_gpu)
    
    from bert_score import score as bert_score
    
    pred_clean = [clean_action_text(p) for p in pred_texts]
    gt_clean = [clean_action_text(g) for g in gt_texts]
    
    # Filter out empty strings
    valid_pairs = [(p, g) for p, g in zip(pred_clean, gt_clean) if p and g]
    
    print(f"Total samples: {len(predictions)}")
    print(f"Valid pairs for BERTScore: {len(valid_pairs)}")
    
    if valid_pairs:
        preds, refs = zip(*valid_pairs)
        
        print(f"Computing BERTScore with microsoft/deberta-v2-xxlarge-mnli...")
        print(f"Using GPU: {bertscore_gpu}, Batch size: {batch_size}")
        
        P, R, F1 = bert_score(
            list(preds), 
            list(refs),
            model_type="microsoft/deberta-v2-xxlarge-mnli",
            lang="en",
            batch_size=batch_size,
            rescale_with_baseline=True,
            verbose=True
        )
        
        metrics['bertscore_precision'] = float(P.mean())
        metrics['bertscore_recall'] = float(R.mean())
        metrics['bertscore_f1'] = float(F1.mean())
        
        print(f"\nBERTScore Results (microsoft/deberta-v2-xxlarge-mnli):")
        print(f"  Precision: {metrics['bertscore_precision']:.6f}")
        print(f"  Recall: {metrics['bertscore_recall']:.6f}")
        print(f"  F1: {metrics['bertscore_f1']:.6f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate BERTScore using deberta-v2-xxlarge-mnli"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        required=True,
        help="Path to results directory containing per_sample_results.json"
    )
    parser.add_argument(
        "--bertscore-gpu", 
        type=int, 
        default=0,
        help="GPU ID for BERTScore computation (default: 0)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for BERTScore (default: 32, reduce if OOM)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_deberta",
        help="Suffix for new metrics file (default: _deberta)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load per_sample_results.json
    per_sample_path = results_dir / "per_sample_results.json"
    if not per_sample_path.exists():
        print(f"Error: {per_sample_path} not found")
        sys.exit(1)
    
    print(f"Loading results from: {per_sample_path}")
    with open(per_sample_path, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} samples")
    
    # Load existing metrics
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            original_metrics = json.load(f)
        print(f"\nOriginal metrics (roberta-large):")
        for k, v in original_metrics.items():
            print(f"  {k}: {v:.6f}")
    else:
        original_metrics = {}
    
    # Compute new BERTScore
    print("\n" + "=" * 60)
    new_bertscore = compute_bertscore_deberta(
        predictions, 
        bertscore_gpu=args.bertscore_gpu,
        batch_size=args.batch_size
    )
    print("=" * 60)
    
    # Create updated metrics
    updated_metrics = original_metrics.copy()
    updated_metrics.update(new_bertscore)
    
    # Save updated metrics
    new_metrics_path = results_dir / f"metrics{args.output_suffix}.json"
    with open(new_metrics_path, 'w') as f:
        json.dump(updated_metrics, f, indent=2)
    print(f"\nUpdated metrics saved to: {new_metrics_path}")
    
    # Also update the original metrics.json if desired
    backup_path = results_dir / "metrics_roberta_backup.json"
    with open(backup_path, 'w') as f:
        json.dump(original_metrics, f, indent=2)
    print(f"Original metrics backed up to: {backup_path}")
    
    with open(metrics_path, 'w') as f:
        json.dump(updated_metrics, f, indent=2)
    print(f"Main metrics.json updated with deberta scores")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison (roberta-large vs deberta-v2-xxlarge-mnli):")
    print("=" * 60)
    for key in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
        old_val = original_metrics.get(key, 0)
        new_val = new_bertscore.get(key, 0)
        diff = new_val - old_val
        print(f"{key}:")
        print(f"  roberta-large:    {old_val:.6f}")
        print(f"  deberta-xxlarge:  {new_val:.6f}")
        print(f"  difference:       {diff:+.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
