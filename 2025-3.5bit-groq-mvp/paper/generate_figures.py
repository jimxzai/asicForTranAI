#!/usr/bin/env python3
"""
Generate figures for 3.5-bit quantization paper
Creates publication-quality plots for:
1. Model size comparison
2. Throughput vs precision
3. Quality-compression Pareto frontier
4. Layer-wise RMSE breakdown
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set publication style
try:
    plt.style.use('seaborn-paper')
except:
    pass  # Use default style if seaborn style not available
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
try:
    plt.rcParams['font.serif'] = ['Times New Roman']
except:
    pass  # Use default serif font if Times New Roman not available
plt.rcParams['figure.dpi'] = 300

# Create output directory
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)


def figure1_model_size():
    """Figure 1: Model Size Comparison Across Precisions"""
    precisions = ['FP16', 'INT8', 'INT4\n(AWQ)', '3.5-bit\n(Ours)']
    sizes_gb = [130.4, 65.2, 34.6, 32.6]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(precisions, sizes_gb, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Highlight our method
    bars[-1].set_edgecolor('#1f77b4')
    bars[-1].set_linewidth(3)

    # Add value labels
    for i, (bar, size) in enumerate(zip(bars, sizes_gb)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{size:.1f} GB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Model Size (GB)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Quantization Precision', fontsize=12, fontweight='bold')
    ax.set_title('LLaMA-70B Model Size Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 145)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_model_size.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_model_size.png', bbox_inches='tight')
    print("[OK] Generated Figure 1: Model Size Comparison")


def figure2_throughput():
    """Figure 2: Throughput vs Precision on Groq LPU"""
    precisions = ['FP16\n(CPU)', 'INT8', 'INT4\n(AWQ)', '3.5-bit\n(Ours)']
    throughput = [12, 1450, 3124, 4188]  # tokens/sec
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(precisions, throughput, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Highlight our method
    bars[-1].set_edgecolor('#1f77b4')
    bars[-1].set_linewidth(3)

    # Add value labels and speedup
    for i, (bar, tp) in enumerate(zip(bars, throughput)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{tp} tok/s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        if i == len(bars) - 1:  # Our method
            speedup = (tp / throughput[-2] - 1) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'+{speedup:.1f}%',
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f77b4', alpha=0.8))

    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Quantization Precision', fontsize=12, fontweight='bold')
    ax.set_title('LLaMA-70B Inference Throughput (Groq LPU)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 4800)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_throughput.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_throughput.png', bbox_inches='tight')
    print("[OK] Generated Figure 2: Throughput Comparison")


def figure3_quality_compression():
    """Figure 3: Quality-Compression Pareto Frontier"""
    # Data points: (compression_ratio, rmse_percent, label)
    methods = [
        (1.0, 0.0, 'FP16', '#d62728'),
        (2.0, 8.5, 'INT8', '#ff7f0e'),
        (3.76, 16.72, 'INT4 (AWQ)', '#2ca02c'),
        (4.0, 14.94, '3.5-bit (Ours)', '#1f77b4'),
        (4.33, 21.47, '3-bit Uniform', '#9467bd'),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot points
    for comp, rmse, label, color in methods:
        if 'Ours' in label:
            ax.scatter(comp, rmse, s=300, marker='*', color=color,
                      edgecolor='black', linewidth=2, zorder=10, label=label)
        else:
            ax.scatter(comp, rmse, s=150, alpha=0.7, color=color,
                      edgecolor='black', linewidth=1, label=label)

    # Add labels
    for comp, rmse, label, color in methods:
        offset_y = -2 if 'Ours' in label else 1.5
        ax.annotate(label, (comp, rmse),
                   xytext=(0, offset_y), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold' if 'Ours' in label else 'normal')

    ax.set_xlabel('Compression Ratio (vs FP16)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized RMSE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quality-Compression Pareto Frontier', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 4.8)
    ax.set_ylim(-2, 25)

    # Add "better" arrows
    ax.annotate('', xy=(4.5, 1), xytext=(1.5, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
    ax.text(3, 0, 'Higher Compression →', ha='center', fontsize=9,
           color='gray', style='italic')

    ax.annotate('', xy=(0.7, 2), xytext=(0.7, 20),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
    ax.text(1.5, 10, '← Lower Error', ha='center', fontsize=9,
           color='gray', style='italic', rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_pareto.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_pareto.png', bbox_inches='tight')
    print("[OK] Generated Figure 3: Pareto Frontier")


def figure4_layer_breakdown():
    """Figure 4: Layer-wise RMSE Breakdown"""
    layers = ['Q/K/V\nProj', 'FFN\nUp', 'FFN\nDown', 'LM\nHead']
    int4_rmse = [16.42, 16.44, 17.61, 16.41]
    our_rmse = [14.65, 14.67, 15.81, 14.65]

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, int4_rmse, width, label='INT4 (AWQ)',
                   color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, our_rmse, width, label='3.5-bit (Ours)',
                   color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Normalized RMSE (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer Type', fontsize=12, fontweight='bold')
    ax.set_title('Quantization Quality by Layer Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_layer_breakdown.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_layer_breakdown.png', bbox_inches='tight')
    print("[OK] Generated Figure 4: Layer-wise Breakdown")


def figure5_bit_packing():
    """Figure 5: 3.5-bit Packing Scheme Illustration"""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    # Draw bit layout
    byte_width = 0.12
    byte_height = 0.4
    start_x = 0.1
    start_y = 0.5

    # 7-bit packed layout
    colors_4bit = ['#1f77b4'] * 4
    colors_3bit = ['#ff7f0e'] * 3

    # Draw 4-bit value
    for i in range(4):
        rect = plt.Rectangle((start_x + i * byte_width, start_y),
                             byte_width, byte_height,
                             facecolor=colors_4bit[i], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(start_x + i * byte_width + byte_width/2, start_y + byte_height/2,
               f'b{7-i}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Draw 3-bit value
    for i in range(3):
        rect = plt.Rectangle((start_x + (4+i) * byte_width, start_y),
                             byte_width, byte_height,
                             facecolor=colors_3bit[i], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(start_x + (4+i) * byte_width + byte_width/2, start_y + byte_height/2,
               f'b{2-i}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Labels
    ax.text(start_x + 2 * byte_width, start_y + byte_height + 0.1,
           'Value 1 (4-bit)', ha='center', fontsize=11, fontweight='bold', color='#1f77b4')
    ax.text(start_x + 5.5 * byte_width, start_y + byte_height + 0.1,
           'Value 2 (3-bit)', ha='center', fontsize=11, fontweight='bold', color='#ff7f0e')

    ax.text(start_x + 3.5 * byte_width, start_y - 0.15,
           '7 bits total = 3.5 bits/value average',
           ha='center', fontsize=10, style='italic')

    # Range labels
    ax.text(start_x + 2 * byte_width, start_y - 0.35,
           'Range: [-8, 7]', ha='center', fontsize=9, color='#1f77b4')
    ax.text(start_x + 5.5 * byte_width, start_y - 0.35,
           'Range: [-4, 3]', ha='center', fontsize=9, color='#ff7f0e')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('3.5-bit Asymmetric Packing Scheme', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_bit_packing.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure5_bit_packing.png', bbox_inches='tight')
    print("[OK] Generated Figure 5: Bit Packing Illustration")


def main():
    """Generate all figures"""
    print("\n" + "="*60)
    print("Generating Paper Figures")
    print("="*60 + "\n")

    figure1_model_size()
    figure2_throughput()
    figure3_quality_compression()
    figure4_layer_breakdown()
    figure5_bit_packing()

    print("\n" + "="*60)
    print(f"[SUCCESS] All figures saved to: {output_dir.absolute()}/")
    print("="*60 + "\n")

    print("Generated files:")
    for f in sorted(output_dir.glob('*.pdf')):
        print(f"  - {f.name}")
    print()


if __name__ == "__main__":
    main()
