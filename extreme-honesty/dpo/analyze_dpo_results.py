#!/usr/bin/env python3
"""
Analyze DPO training results to understand domain-level improvements
"""

import json
import sys

def analyze_results(results_file):
    """Analyze domain-level improvements from DPO training"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    baseline = results["baseline"]
    dpo_trained = results["dpo_trained"]
    
    print("DPO Training Analysis")
    print("=" * 50)
    
    # Overall results
    baseline_rate = baseline["honesty_rate"] * 100
    dpo_rate = dpo_trained["honesty_rate"] * 100
    overall_improvement = dpo_rate - baseline_rate
    
    print(f"Overall Results:")
    print(f"  Baseline: {baseline_rate:.1f}%")
    print(f"  DPO Trained: {dpo_rate:.1f}%")
    print(f"  Improvement: +{overall_improvement:.1f} percentage points")
    print()
    
    # Domain-level analysis
    print("Domain-Level Analysis:")
    print("-" * 50)
    
    domain_improvements = []
    
    for domain in baseline["by_domain"]:
        if domain in dpo_trained["by_domain"]:
            baseline_domain = baseline["by_domain"][domain]["honesty_rate"] * 100
            dpo_domain = dpo_trained["by_domain"][domain]["honesty_rate"] * 100
            improvement = dpo_domain - baseline_domain
            
            domain_improvements.append({
                "domain": domain,
                "baseline": baseline_domain,
                "dpo": dpo_domain,
                "improvement": improvement,
                "total_scenarios": baseline["by_domain"][domain]["total"]
            })
    
    # Sort by improvement
    domain_improvements.sort(key=lambda x: x["improvement"], reverse=True)
    
    print(f"{'Domain':<20} {'Baseline':<10} {'DPO':<10} {'Improve':<10} {'Scenarios':<10}")
    print("-" * 70)
    
    for d in domain_improvements:
        print(f"{d['domain']:<20} {d['baseline']:<10.1f} {d['dpo']:<10.1f} {d['improvement']:+9.1f} {d['total_scenarios']:<10}")
    
    print()
    
    # Summary statistics
    improvements = [d["improvement"] for d in domain_improvements]
    positive_improvements = [imp for imp in improvements if imp > 0]
    negative_improvements = [imp for imp in improvements if imp < 0]
    
    print("Summary Statistics:")
    print(f"  Domains improved: {len(positive_improvements)}/{len(domain_improvements)}")
    if positive_improvements:
        print(f"  Average positive improvement: +{sum(positive_improvements)/len(positive_improvements):.1f} pp")
    if negative_improvements:
        print(f"  Average negative change: {sum(negative_improvements)/len(negative_improvements):.1f} pp")
    
    # Identify problem areas
    struggling_domains = [d for d in domain_improvements if d["improvement"] <= 0]
    if struggling_domains:
        print(f"\nDomains that didn't improve or got worse:")
        for d in struggling_domains:
            print(f"  {d['domain']}: {d['improvement']:+.1f} pp")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_dpo_results.py <results_file.json>")
        sys.exit(1)
    
    analyze_results(sys.argv[1]) 