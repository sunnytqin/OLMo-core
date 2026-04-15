#!/usr/bin/env python3
"""
Generate combined heatmap figures: one per model size.
Rows = chinchilla multiplier (fresh data size), columns = epoch.
Each subplot has its own colorbar for local optimal visibility.

Usage:
    python chinchilla_heatmaps.py [--merged-dir ../results/hparam/merged] [--output-dir ../results/hparam/heatmaps]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Hyperparameter grids per model size
GRIDS = {
    "30M": {
        "weight_decays": [0.1, 0.2, 0.4, 0.8, 1.6],
        "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3],
    },
    "60M": {
        "weight_decays": [0.1, 0.2, 0.4, 0.8, 1.6],
        "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3],
    },
    "190M": {
        "weight_decays": [0.1, 0.2, 0.4, 0.8, 1.6],
        "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3],
    },
    "370M": {
        "weight_decays": [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
        "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3, 6e-3],
    },
}

# Chinchilla multiplier extracted from dir name, used for row ordering
CHIN_ORDER = [0.05, 0.1, 0.25, 0.5, 1, 2, 4]

RUN_PATTERN = re.compile(
    r'(30M|60M|190M|370M)_seed\d+_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)'
)


def parse_run_name(run_name: str):
    m = RUN_PATTERN.search(run_name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))


def parse_merged_filename(stem: str):
    for size in ["370M", "60M", "30M"]:
        suffix = f"_{size}"
        if stem.endswith(suffix):
            chin = stem[:-len(suffix)]
            return chin, size
    return None, None


def chin_scale(chin_dir: str) -> float:
    return float(chin_dir.replace("chinchilla_", ""))


def load_all_data(merged_dir: Path):
    """Returns dict of (chin_dir, model_size, epoch) -> {(wd, lr): val_loss}"""
    data = {}
    for json_file in sorted(merged_dir.glob("*.json")):
        chin, size = parse_merged_filename(json_file.stem)
        if chin is None:
            continue
        with open(json_file) as f:
            results = json.load(f)
        epoch_data = defaultdict(dict)
        for run_name, metrics in results.items():
            if "error" in metrics:
                continue
            parsed = parse_run_name(run_name)
            if parsed is None:
                continue
            _, epoch, wd, lr = parsed
            val_loss = metrics.get("validation_loss")
            if val_loss is not None:
                epoch_data[epoch][(wd, lr)] = val_loss
        for epoch, grid_data in epoch_data.items():
            data[(chin, size, epoch)] = grid_data
    return data


def generate_combined_figure(data, model_size: str, output_path: str):
    grid = GRIDS[model_size]
    weight_decays = grid["weight_decays"]
    learning_rates = grid["learning_rates"]
    n_wd = len(weight_decays)
    n_lr = len(learning_rates)

    # Find all chinchilla scales and epochs for this model size
    chin_dirs = sorted(
        set(chin for chin, size, _ in data if size == model_size),
        key=chin_scale
    )
    epochs = sorted(set(epoch for _, size, epoch in data if size == model_size))

    if not chin_dirs or not epochs:
        print(f"  No data for {model_size}, skipping")
        return

    n_rows = len(chin_dirs)
    n_cols = len(epochs)

    cell_w = 2.8 if model_size == "30M" else 3.5
    cell_h = 2.2 if model_size == "30M" else 2.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w + 1.5, n_rows * cell_h + 1.5),
        squeeze=False,
    )

    for r, chin in enumerate(chin_dirs):
        for c, epoch in enumerate(epochs):
            ax = axes[r][c]
            key = (chin, model_size, epoch)

            if key not in data or not data[key]:
                ax.set_visible(False)
                continue

            grid_data = data[key]
            loss_grid = np.full((n_wd, n_lr), np.nan)
            for i, wd in enumerate(weight_decays):
                for j, lr in enumerate(learning_rates):
                    if (wd, lr) in grid_data:
                        loss_grid[i, j] = grid_data[(wd, lr)]

            mask = np.isnan(loss_grid)

            # Local colorbar per subplot
            valid = loss_grid[~mask]
            if len(valid) == 0:
                ax.set_visible(False)
                continue

            sns.heatmap(
                loss_grid,
                annot=True,
                fmt='.2f',
                cmap="viridis_r",
                cbar=False,
                xticklabels=[f'{lr:.0e}' for lr in learning_rates],
                yticklabels=[f'{wd}' for wd in weight_decays],
                ax=ax,
                mask=mask,
                square=False,
                annot_kws={'fontsize': 6},
            )

            # Highlight the best (lowest loss) cell
            best_val = np.nanmin(loss_grid)
            best_idx = np.unravel_index(np.nanargmin(loss_grid), loss_grid.shape)
            ax.add_patch(plt.Rectangle(
                (best_idx[1], best_idx[0]), 1, 1,
                fill=False, edgecolor='red', linewidth=2.5
            ))

            chin_label = chin.replace("chinchilla_", "")
            ax.set_title(f'{chin_label}x / ep{epoch}', fontsize=8, fontweight='bold')

            if c == 0:
                ax.set_ylabel(f'WD', fontsize=7)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            if r == n_rows - 1:
                ax.set_xlabel('LR', fontsize=7)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            ax.tick_params(labelsize=6)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Row labels on the left
    for r, chin in enumerate(chin_dirs):
        chin_label = chin.replace("chinchilla_", "")
        axes[r][0].annotate(
            f'{chin_label}x',
            xy=(-0.4, 0.5), xycoords='axes fraction',
            fontsize=10, fontweight='bold', ha='center', va='center',
            rotation=90,
        )

    # Column labels on top
    for c, epoch in enumerate(epochs):
        axes[0][c].annotate(
            f'Epoch {epoch}',
            xy=(0.5, 1.25), xycoords='axes fraction',
            fontsize=9, fontweight='bold', ha='center', va='bottom',
        )

    fig.suptitle(
        f'{model_size} — Validation Loss (WD vs LR)\n'
        f'Rows: Chinchilla multiplier (fresh data size)  |  Columns: Epochs\n'
        f'Red box = best hyperparams per setting',
        fontsize=12, fontweight='bold', y=1.02,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate combined heatmap figures")
    parser.add_argument("--merged-dir", type=str, default="merged")
    parser.add_argument("--output-dir", type=str, default="heatmaps")
    args = parser.parse_args()

    merged_path = Path(args.merged_dir)
    output_path = Path(args.output_dir)

    print("Loading all merged results...")
    data = load_all_data(merged_path)
    print(f"Found {len(data)} (chinchilla, size, epoch) combinations")

    for model_size in ["30M", "60M", "190M", "370M"]:
        print(f"\nGenerating {model_size} figure...")
        out_file = output_path / f"{model_size}_heatmap_grid.pdf"
        generate_combined_figure(data, model_size, str(out_file))

    print("\nDone!")


if __name__ == "__main__":
    main()
