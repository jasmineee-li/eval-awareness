#!/usr/bin/env python3
"""Plot consistency analysis results for SFT and DPO training."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colors
COLORS = {
    'base': '#6c757d',      # gray
    'sft': '#28a745',       # green
    'dpo': '#dc3545',       # red
}

def load_results():
    """Load SFT and DPO results."""
    base_dir = Path(__file__).parent.parent / "data" / "inference_results"
    
    with open(base_dir / "consistency_analysis.json") as f:
        sft_data = json.load(f)
    
    with open(base_dir / "consistency_analysis_dpo.json") as f:
        dpo_data = json.load(f)
    
    return {
        'base': sft_data['base'],
        'sft': sft_data['lora'],
        'dpo': dpo_data['lora'],
    }


def plot_comparison(results, output_dir: Path):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Eval-Awareness Consistency Training Results\n(QwQ-32B on 1,663 test samples)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    models = ['Base', 'SFT', 'DPO']
    x = np.arange(len(models))
    width = 0.6
    
    # --- Plot 1: Edit Distance (lower is better) ---
    ax1 = axes[0]
    edit_means = [
        results['base']['edit_distance']['mean'],
        results['sft']['edit_distance']['mean'],
        results['dpo']['edit_distance']['mean'],
    ]
    edit_stds = [
        results['base']['edit_distance']['std'],
        results['sft']['edit_distance']['std'],
        results['dpo']['edit_distance']['std'],
    ]
    
    bars1 = ax1.bar(x, edit_means, width, yerr=edit_stds, capsize=5,
                    color=[COLORS['base'], COLORS['sft'], COLORS['dpo']],
                    edgecolor='black', linewidth=1.2, alpha=0.85)
    
    ax1.set_ylabel('Edit Distance (normalized)')
    ax1.set_title('Edit Distance\n(lower = more consistent)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0.6, 0.8)
    
    # Add value labels
    for bar, val in zip(bars1, edit_means):
        ax1.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    # Add delta annotations
    base_ed = edit_means[0]
    for i, (model, val) in enumerate(zip(models[1:], edit_means[1:]), 1):
        delta = val - base_ed
        delta_pct = (delta / base_ed) * 100
        color = COLORS['sft'] if delta < 0 else COLORS['dpo']
        symbol = '↓' if delta < 0 else '↑'
        ax1.annotate(f'{symbol} {abs(delta_pct):.1f}%', xy=(x[i], 0.62),
                     ha='center', fontsize=9, color=color, fontweight='bold')
    
    # --- Plot 2: Embedding Similarity (higher is better) ---
    ax2 = axes[1]
    embed_means = [
        results['base']['embedding_similarity']['mean'],
        results['sft']['embedding_similarity']['mean'],
        results['dpo']['embedding_similarity']['mean'],
    ]
    embed_stds = [
        results['base']['embedding_similarity']['std'],
        results['sft']['embedding_similarity']['std'],
        results['dpo']['embedding_similarity']['std'],
    ]
    
    bars2 = ax2.bar(x, embed_means, width, yerr=embed_stds, capsize=5,
                    color=[COLORS['base'], COLORS['sft'], COLORS['dpo']],
                    edgecolor='black', linewidth=1.2, alpha=0.85)
    
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Embedding Similarity\n(higher = more consistent)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0.75, 0.95)
    
    for bar, val in zip(bars2, embed_means):
        ax2.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    base_es = embed_means[0]
    for i, (model, val) in enumerate(zip(models[1:], embed_means[1:]), 1):
        delta = val - base_es
        delta_pct = (delta / base_es) * 100
        color = COLORS['sft'] if delta > 0 else COLORS['dpo']
        symbol = '↑' if delta > 0 else '↓'
        ax2.annotate(f'{symbol} {abs(delta_pct):.1f}%', xy=(x[i], 0.77),
                     ha='center', fontsize=9, color=color, fontweight='bold')
    
    # --- Plot 3: Length Ratio (closer to 1.0 is better) ---
    ax3 = axes[2]
    len_means = [
        results['base']['length_ratio']['mean'],
        results['sft']['length_ratio']['mean'],
        results['dpo']['length_ratio']['mean'],
    ]
    len_stds = [
        results['base']['length_ratio']['std'],
        results['sft']['length_ratio']['std'],
        results['dpo']['length_ratio']['std'],
    ]
    
    bars3 = ax3.bar(x, len_means, width, yerr=len_stds, capsize=5,
                    color=[COLORS['base'], COLORS['sft'], COLORS['dpo']],
                    edgecolor='black', linewidth=1.2, alpha=0.85)
    
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (1.0)')
    ax3.set_ylabel('Length Ratio (prefixed / non-prefixed)')
    ax3.set_title('Response Length Ratio\n(1.0 = no change)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.set_ylim(0.5, 1.6)
    ax3.legend(loc='upper right', fontsize=9)
    
    for bar, val in zip(bars3, len_means):
        ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "consistency_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    output_pdf = output_dir / "consistency_comparison.pdf"
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_pdf}")
    
    plt.close()


def plot_detailed_comparison(results, output_dir: Path):
    """Create detailed box-plot style comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Edit Distance Distribution Comparison', fontsize=14, fontweight='bold')
    
    models = ['Base', 'SFT', 'DPO']
    x = np.arange(len(models))
    
    # Edit distance with percentiles
    ax1 = axes[0]
    
    ed_data = [
        (results['base']['edit_distance']['median'], 
         results['base']['edit_distance']['p25'],
         results['base']['edit_distance']['p75']),
        (results['sft']['edit_distance']['median'],
         results['sft']['edit_distance']['p25'],
         results['sft']['edit_distance']['p75']),
        (results['dpo']['edit_distance']['median'],
         results['dpo']['edit_distance']['p25'],
         results['dpo']['edit_distance']['p75']),
    ]
    
    medians = [d[0] for d in ed_data]
    p25s = [d[1] for d in ed_data]
    p75s = [d[2] for d in ed_data]
    
    # IQR bars
    for i, (med, p25, p75) in enumerate(ed_data):
        color = [COLORS['base'], COLORS['sft'], COLORS['dpo']][i]
        ax1.bar(x[i], p75 - p25, bottom=p25, width=0.5, color=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax1.hlines(med, x[i]-0.25, x[i]+0.25, colors=color, linewidth=3)
        ax1.scatter(x[i], med, color=color, s=100, zorder=5, edgecolor='black')
    
    ax1.set_ylabel('Edit Distance')
    ax1.set_title('Median & IQR (25th-75th percentile)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0.6, 0.8)
    
    # Summary table
    ax2 = axes[1]
    ax2.axis('off')
    
    table_data = [
        ['Metric', 'Base', 'SFT', 'DPO', 'Best'],
        ['Edit Dist. (mean)', f"{results['base']['edit_distance']['mean']:.4f}",
         f"{results['sft']['edit_distance']['mean']:.4f}",
         f"{results['dpo']['edit_distance']['mean']:.4f}",
         'SFT ✓' if results['sft']['edit_distance']['mean'] < results['dpo']['edit_distance']['mean'] else 'DPO'],
        ['Edit Dist. (med)', f"{results['base']['edit_distance']['median']:.4f}",
         f"{results['sft']['edit_distance']['median']:.4f}",
         f"{results['dpo']['edit_distance']['median']:.4f}",
         'SFT ✓' if results['sft']['edit_distance']['median'] < results['dpo']['edit_distance']['median'] else 'DPO'],
        ['Embed Sim. (mean)', f"{results['base']['embedding_similarity']['mean']:.4f}",
         f"{results['sft']['embedding_similarity']['mean']:.4f}",
         f"{results['dpo']['embedding_similarity']['mean']:.4f}",
         'SFT ✓' if results['sft']['embedding_similarity']['mean'] > results['dpo']['embedding_similarity']['mean'] else 'DPO'],
        ['Length Ratio', f"{results['base']['length_ratio']['mean']:.3f}",
         f"{results['sft']['length_ratio']['mean']:.3f}",
         f"{results['dpo']['length_ratio']['mean']:.3f}",
         'SFT ✓' if abs(results['sft']['length_ratio']['mean'] - 1) < abs(results['dpo']['length_ratio']['mean'] - 1) else 'DPO'],
    ]
    
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor('#343a40')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best column
    for i in range(1, 5):
        table[(i, 4)].set_facecolor('#d4edda')
    
    ax2.set_title('Summary Table', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = output_dir / "consistency_detailed.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.close()


def print_summary(results):
    """Print text summary."""
    print("\n" + "="*70)
    print("CONSISTENCY TRAINING RESULTS SUMMARY")
    print("="*70)
    
    base_ed = results['base']['edit_distance']['mean']
    sft_ed = results['sft']['edit_distance']['mean']
    dpo_ed = results['dpo']['edit_distance']['mean']
    
    base_es = results['base']['embedding_similarity']['mean']
    sft_es = results['sft']['embedding_similarity']['mean']
    dpo_es = results['dpo']['embedding_similarity']['mean']
    
    print(f"\n{'Model':<10} {'Edit Dist':<12} {'Δ vs Base':<12} {'Embed Sim':<12} {'Δ vs Base':<12}")
    print("-"*60)
    print(f"{'Base':<10} {base_ed:<12.4f} {'-':<12} {base_es:<12.4f} {'-':<12}")
    print(f"{'SFT':<10} {sft_ed:<12.4f} {(sft_ed-base_ed)/base_ed*100:+.2f}%{'':<6} {sft_es:<12.4f} {(sft_es-base_es)/base_es*100:+.2f}%")
    print(f"{'DPO':<10} {dpo_ed:<12.4f} {(dpo_ed-base_ed)/base_ed*100:+.2f}%{'':<6} {dpo_es:<12.4f} {(dpo_es-base_es)/base_es*100:+.2f}%")
    
    print("\n" + "-"*60)
    print("VERDICT:")
    print(f"  • SFT: {'✓ Improved' if sft_ed < base_ed else '✗ Worse'} (edit dist {(sft_ed-base_ed)/base_ed*100:+.2f}%)")
    print(f"  • DPO: {'✓ Improved' if dpo_ed < base_ed else '✗ Worse'} (edit dist {(dpo_ed-base_ed)/base_ed*100:+.2f}%)")
    print(f"  • Winner: {'SFT' if sft_ed < dpo_ed else 'DPO'}")
    print("="*70 + "\n")


def main():
    output_dir = Path(__file__).parent.parent / "data" / "inference_results"
    
    print("Loading results...")
    results = load_results()
    
    print_summary(results)
    
    print("Generating plots...")
    plot_comparison(results, output_dir)
    plot_detailed_comparison(results, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
