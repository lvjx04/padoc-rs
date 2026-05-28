#!/usr/bin/env python3
"""
Generate 4 separate piecewise-linear ts figures (one per dataset).

Each figure shows 3-4 representative GPU kernels from one dataset,
demonstrating that timestamps grow piecewise-linearly with instance index.

Input:  CSV files in /tmp/ts_extract/ produced by extract_all_datasets_ts.rs
Output: results/figures/ts_linear_{dataset}.png (and .pdf)

Usage:
  python3 scripts/plot_ts_linear_4datasets.py [output_dir]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Paper-quality settings
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DATA_DIR = '/tmp/ts_extract'


def human_ts(x, pos):
    """Format timestamps as human-readable."""
    if abs(x) >= 1e9:
        return f'{x/1e9:.1f}B'
    elif abs(x) >= 1e6:
        return f'{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


def load_csv(path, max_points=3000):
    """Load index,ts CSV file, subsampling if too large."""
    data = np.loadtxt(path, delimiter=',', dtype=np.int64)
    if data.ndim == 1:
        return data[0:1], data[1:2]
    idx, ts = data[:, 0], data[:, 1]
    if len(idx) > max_points:
        step = len(idx) / max_points
        sel = np.array([int(i * step) for i in range(max_points)])
        idx, ts = idx[sel], ts[sel]
    return idx, ts


def shorten_name(name):
    """Shorten long kernel names for display."""
    # Remove common prefixes/suffixes
    short = name
    if 'void at::native::' in short:
        short = short.replace('void at::native::', '')
    if 'elementwise_kernel' in short:
        if 'direct_copy' in short:
            return 'elementwise (copy)'
        elif 'CUDAFunctor_add' in short:
            return 'elementwise (add)'
        elif 'silu_kernel' in short:
            return 'elementwise (SiLU)'
        elif 'GeluCUDA' in short:
            return 'elementwise (GeLU)'
        else:
            return 'elementwise_kernel'
    if 'vectorized_layer_norm' in short:
        return 'LayerNorm'
    if 'Cijk_Alik_Bljk' in short:
        return 'GEMM (rocBLAS)'
    if 'fused_merge_kernel' in short:
        return 'fused_merge_kernel'
    if 'Gemm_tcu_mr_kernel' in short:
        return 'GEMM (TCU)'
    if 'genericOp' in short or 'genericMultiShmOp' in short:
        return 'NCCL AllReduce'
    if len(short) > 30:
        return short[:27] + '...'
    return short


def plot_dataset(ax, dataset_name, files_and_labels, colors, title_extra=''):
    """Plot multiple kernels on one axis."""
    for i, (csv_file, label) in enumerate(files_and_labels):
        path = os.path.join(DATA_DIR, csv_file)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        idx, ts = load_csv(path)
        n = len(idx)
        display_label = f'{shorten_name(label)} ({n:,})'
        ax.scatter(idx, ts, s=0.5, alpha=0.6, color=colors[i % len(colors)],
                   label=display_label, rasterized=True)

    ax.set_xlabel('Instance Index', fontsize=9)
    ax.set_ylabel('Timestamp (us)', fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(human_ts))
    ax.legend(fontsize=7, loc='upper left', markerscale=5, framealpha=0.8)
    ax.set_title(f'{dataset_name}{title_extra}', fontsize=10, fontweight='bold')


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Dataset configurations: (dataset_name, [(csv_file, kernel_label), ...], extra_title)
    datasets = [
        ('LeWorldModel', [
            ('lewm_fused_merge_kernel.csv', 'fused_merge_kernel'),
            ('lewm_layer_norm.csv', 'LayerNorm'),
            ('lewm_Cijk.csv', 'GEMM (rocBLAS)'),
            ('lewm_elementwise.csv', 'elementwise (copy)'),
        ], ' (AMD MI250, Inference)'),

        ('Qwen3', [
            ('qwen3_fwdbwd.csv', 'fwdbwd'),
            ('qwen3_suLaunchKernel.csv', 'suLaunchKernel'),
            ('qwen3_memcpy_dtoh.csv', 'Memcpy DtoH'),
            ('qwen3_copy_kernel.csv', 'CopyKernel'),
        ], ' (Ascend NPU, Training)'),

        ('UniFolm', [
            ('unifolm_conv.csv', 'conv'),
            ('unifolm_gemm.csv', 'gemm'),
            ('unifolm_elementwise.csv', 'elementwise (copy)'),
        ], ' (NVIDIA A100, Training)'),

        ('LLaMA-70B', [
            ('llama_gemm_tcu.csv', 'GEMM (TCU)'),
            ('llama_elementwise_add.csv', 'elementwise (add)'),
            ('llama_nccl.csv', 'NCCL AllReduce'),
            ('llama_colparallel.csv', 'ColumnParallelLinear'),
        ], ' (NVIDIA GPU, Training)'),
    ]

    for dataset_name, files_and_labels, title_extra in datasets:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
        plot_dataset(ax, dataset_name, files_and_labels, colors, title_extra)
        plt.tight_layout()

        out_base = f'ts_linear_{dataset_name.lower().replace("-", "").replace(" ", "_")}'
        png_path = os.path.join(output_dir, f'{out_base}.png')
        pdf_path = os.path.join(output_dir, f'{out_base}.pdf')
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        print(f'Saved: {png_path}')
        plt.close()

    print('Done! Generated 4 figures.')


if __name__ == '__main__':
    main()
