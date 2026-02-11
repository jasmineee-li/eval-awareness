#!/usr/bin/env python3
"""
Merge baseline evaluation results with new DPO evaluation results
================================================================
Useful when you want to reuse baseline evaluations across multiple DPO experiments.

Usage:
    python merge_evaluation_results.py \
        --baseline-results path/to/previous_baseline_results.json \
        --dpo-results path/to/new_dpo_results.json \
        --output path/to/merged_results.json
"""

import argparse
import json
import os

def merge_evaluation_results(baseline_file: str, dpo_file: str, output_file: str):
    """Merge baseline and DPO evaluation results into a single file"""
    
    # Load previous baseline results
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    # Load new DPO results (skip-baseline-eval format)
    with open(dpo_file, 'r') as f:
        dpo_data = json.load(f)
    
    # Merge the results
    merged_results = {
        "baseline": baseline_data.get("baseline", baseline_data),  # Handle both formats
        "dpo_trained": dpo_data.get("dpo_trained", dpo_data)
    }
    
    # Calculate improvement if we have both results
    if "baseline" in merged_results and "dpo_trained" in merged_results:
        baseline_rate = merged_results["baseline"]["honesty_rate"] * 100
        dpo_rate = merged_results["dpo_trained"]["honesty_rate"] * 100
        improvement = dpo_rate - baseline_rate
        
        print(f"Baseline honesty rate: {baseline_rate:.1f}%")
        print(f"DPO-trained honesty rate: {dpo_rate:.1f}%")
        print(f"Improvement: +{improvement:.1f} percentage points")
    
    # Save merged results
    with open(output_file, 'w') as f:
        json.dump(merged_results, f, indent=2)
    
    print(f"Merged results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge baseline and DPO evaluation results")
    parser.add_argument("--baseline-results", required=True, help="Path to baseline evaluation results JSON")
    parser.add_argument("--dpo-results", required=True, help="Path to DPO evaluation results JSON")
    parser.add_argument("--output", required=True, help="Path to save merged results")
    
    args = parser.parse_args()
    
    merge_evaluation_results(args.baseline_results, args.dpo_results, args.output) 