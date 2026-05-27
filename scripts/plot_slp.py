#!/usr/bin/env python3
"""
Plot ts-index linearity and SLP compression effect for the paper.

Reads CSV from export_ts_for_plot example.
Produces two figures:
  1. ts_linearity.pdf  — ts vs instance_index for several templates (shows linearity)
  2. slp_effect.pdf    — SLP segment visualization (shows compression)

Usage:
  python3 scripts/plot_slp.py /tmp/ts_data.csv [output_dir]
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Paper-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def human_readable(x, pos):
    if x >= 1e9:
        return f'{x/1e9:.1f}G'
    elif x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


def plot_ts_linearity(df, output_dir):
    """Figure 1: ts vs index showing near-perfect linearity within templates."""
    templates = df['template_id'].unique()
    n_templates = min(len(templates), 4)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    for i, tmpl_id in enumerate(templates[:n_templates]):
        ax = axes[i]
        sub = df[df['template_id'] == tmpl_id].copy()
        name = sub['template_name'].iloc[0]
        n_inst = sub['instance_count'].iloc[0]

        x = sub['index'].values
        y = sub['ts_value'].values

        # Plot actual ts values
        ax.scatter(x, y, s=1, alpha=0.6, color=colors[i], label='ts values')

        # Plot linear regression fit to show linearity
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            y_fit = np.polyval(coeffs, x)
            ax.plot(x, y_fit, 'r--', linewidth=1.5, alpha=0.8,
                    label=f'linear fit (R²={r_squared(x, y):.6f})')

        # Truncate name for display
        display_name = name if len(name) <= 30 else name[:27] + '...'
        ax.set_title(f'{display_name}\n({n_inst:,} instances)', fontsize=9)
        ax.set_xlabel('Instance index')
        ax.set_ylabel('Timestamp (μs)')
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable))
        ax.legend(fontsize=7, loc='upper left')

    # Hide unused subplots
    for j in range(n_templates, 4):
        axes[j].set_visible(False)

    fig.suptitle('Timestamp vs. Instance Index (per-template)', fontsize=11, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'ts_linearity.png')
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close()


def plot_slp_effect(df, output_dir):
    """Figure 2: SLP compression — segment boundaries + compression stats."""
    templates = df['template_id'].unique()
    # Pick one template with many instances for detailed view
    tmpl_id = templates[0]
    sub = df[df['template_id'] == tmpl_id].copy()
    name = sub['template_name'].iloc[0]
    n_inst = sub['instance_count'].iloc[0]

    x = sub['index'].values
    y = sub['ts_value'].values
    seg_ids = sub['segment_id'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), height_ratios=[2, 1])

    # Top: ts values colored by segment
    unique_segs = np.unique(seg_ids)
    n_segs = len(unique_segs)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_segs, 10)))

    for idx, seg_id in enumerate(unique_segs[:50]):  # Show first 50 segments
        mask = seg_ids == seg_id
        color = colors[idx % len(colors)]
        ax1.plot(x[mask], y[mask], '-', color=color, linewidth=0.8, alpha=0.7)

    ax1.set_xlabel('Instance index')
    ax1.set_ylabel('Timestamp (μs)')
    ax1.yaxis.set_major_formatter(FuncFormatter(human_readable))
    ax1.set_title(
        f'SLP Segmentation: "{name[:35]}..." ({n_inst:,} instances → {n_segs} segments)',
        fontsize=9
    )

    # Bottom: segment length histogram
    # Compute segment lengths from the data
    seg_lengths = []
    for seg_id in unique_segs:
        mask = seg_ids == seg_id
        seg_lengths.append(mask.sum())

    ax2.hist(seg_lengths, bins=min(50, len(seg_lengths)), color='steelblue',
             edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Segment length (instances per segment)')
    ax2.set_ylabel('Count')
    ax2.set_title(
        f'Segment Length Distribution (compression: {n_inst:,} instances → {n_segs} segments = {n_inst/max(n_segs,1):.0f}× avg)',
        fontsize=9
    )
    ax2.axvline(np.mean(seg_lengths), color='red', linestyle='--', linewidth=1,
                label=f'mean = {np.mean(seg_lengths):.1f}')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'slp_effect.png')
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close()


def r_squared(x, y):
    """Compute R² for linear fit."""
    if len(x) < 2:
        return 0.0
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_slp.py <ts_data.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    os.makedirs(output_dir, exist_ok=True)

    # Read CSV, skip lines that look like stderr (start with spaces or '[')
    lines = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith('template_id') or (line[0:1].isdigit()):
                lines.append(line)

    # Write cleaned data to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp.write('template_id,template_name,instance_count,index,ts_value,fitted_value,residual,segment_id\n')
        for line in lines:
            if not line.startswith('template_id'):
                tmp.write(line)
        tmp_path = tmp.name

    df = pd.read_csv(tmp_path)
    os.unlink(tmp_path)

    print(f"Loaded {len(df)} rows, {df['template_id'].nunique()} templates")

    plot_ts_linearity(df, output_dir)
    plot_slp_effect(df, output_dir)
    print("Done!")


if __name__ == '__main__':
    main()
