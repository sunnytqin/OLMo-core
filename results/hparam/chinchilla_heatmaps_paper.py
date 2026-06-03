#!/usr/bin/env python3
"""
Paper-appendix version of the multi_epoch hparam heatmap grid.

Subset of `chinchilla_heatmaps.py`:
  - rows: the first two chinchilla multipliers (0.05, 0.1), relabeled as
    TTP = chin * 20 (so TTP=1 and TTP=2)
  - columns: epochs {1, 4, 16, 64}
  - each cell: WD vs LR validation-loss heatmap with red box on best

Usage:
    python chinchilla_heatmaps_paper.py            # defaults to 370M
    python chinchilla_heatmaps_paper.py --size 30M
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MERGED_DIR = Path(__file__).parent / "merged" / "multi_epoch"
OUTPUT_DIR = Path(__file__).parent / "heatmaps" / "multi_epoch"

CHIN_VALUES = [0.05, 0.1]            # rows
EPOCH_VALUES = [1, 4, 16, 64]        # cols
TTP_PER_CHIN = 20                    # 1 chinchilla = 20 tokens / param

WEIGHT_DECAYS = [0.1, 0.2, 0.4, 0.8, 1.6]
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]

RUN_PATTERN = re.compile(
    r'(14M|30M|60M|190M|370M)_seed\d+_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)'
)


def chin_scale(chin_dir: str) -> float:
    return float(chin_dir.replace("chinchilla_", ""))


def load_data(model_size):
    """(chin_value, epoch) -> {(wd, lr): val_loss} for `model_size`."""
    out = {}
    for chin_value in CHIN_VALUES:
        # JSON files use the literal chinchilla_<num>_<size>.json layout.
        # The number can be e.g. "0.05" or "1" — match the file by parsing.
        match = None
        for f in MERGED_DIR.glob(f"chinchilla_*_{model_size}.json"):
            stem_chin = f.stem[:-len(f"_{model_size}")]
            if abs(chin_scale(stem_chin) - chin_value) < 1e-9:
                match = f
                break
        if match is None:
            print(f"  WARN: no merged json found for chin={chin_value}")
            continue
        with open(match) as fh:
            results = json.load(fh)
        per_epoch = defaultdict(dict)
        for run_name, metrics in results.items():
            if "error" in metrics:
                continue
            m = RUN_PATTERN.search(run_name)
            if not m:
                continue
            _, ep, wd, lr = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
            val_loss = metrics.get("validation_loss")
            if val_loss is not None:
                per_epoch[ep][(wd, lr)] = val_loss
        for ep in EPOCH_VALUES:
            out[(chin_value, ep)] = per_epoch.get(ep, {})
    return out


def make_figure(data, output_path, model_size):
    n_rows = len(CHIN_VALUES)
    n_cols = len(EPOCH_VALUES)
    n_wd = len(WEIGHT_DECAYS)
    n_lr = len(LEARNING_RATES)

    cell_w, cell_h = 2.6, 2.2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w + 1.0, n_rows * cell_h + 0.8),
        squeeze=False,
    )

    for r, chin in enumerate(CHIN_VALUES):
        ttp = int(round(chin * TTP_PER_CHIN))
        for c, ep in enumerate(EPOCH_VALUES):
            ax = axes[r][c]
            grid_data = data.get((chin, ep), {})

            loss_grid = np.full((n_wd, n_lr), np.nan)
            for i, wd in enumerate(WEIGHT_DECAYS):
                for j, lr in enumerate(LEARNING_RATES):
                    if (wd, lr) in grid_data:
                        loss_grid[i, j] = grid_data[(wd, lr)]
            mask = np.isnan(loss_grid)

            if mask.all():
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=10, color="#888888", transform=ax.transAxes)
            else:
                sns.heatmap(
                    loss_grid,
                    annot=True,
                    fmt='.2f',
                    cmap="viridis_r",
                    cbar=False,
                    xticklabels=[f'{lr:.0e}' for lr in LEARNING_RATES],
                    yticklabels=[f'{wd}' for wd in WEIGHT_DECAYS],
                    ax=ax,
                    mask=mask,
                    square=False,
                    annot_kws={'fontsize': 8},
                )
                best_idx = np.unravel_index(np.nanargmin(loss_grid), loss_grid.shape)
                ax.add_patch(plt.Rectangle(
                    (best_idx[1], best_idx[0]), 1, 1,
                    fill=False, edgecolor='red', linewidth=2.5,
                ))

            # Per-cell title sits inside the column; we replace it with
            # outer row/column annotations below for a cleaner look.
            ax.set_title('')

            if c == 0:
                ax.set_ylabel('WD', fontsize=9)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            if r == n_rows - 1:
                ax.set_xlabel('LR', fontsize=9)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            ax.tick_params(labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Row labels: TTP value on the left side of row 0's first column.
    for r, chin in enumerate(CHIN_VALUES):
        ttp = int(round(chin * TTP_PER_CHIN))
        axes[r][0].annotate(
            f'TTP = {ttp}',
            xy=(-0.45, 0.5), xycoords='axes fraction',
            fontsize=12, fontweight='bold', ha='center', va='center',
            rotation=90,
        )

    # Column labels: epoch on top of each column.
    for c, ep in enumerate(EPOCH_VALUES):
        axes[0][c].annotate(
            f'Epochs = {ep}',
            xy=(0.5, 1.12), xycoords='axes fraction',
            fontsize=11, fontweight='bold', ha='center', va='bottom',
        )

    fig.suptitle(
        f'{model_size} — Validation loss across (WD, LR) per (TTP, Epochs).  '
        'Red box marks the optimum per cell.',
        fontsize=11, y=1.02,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="370M",
                        choices=["14M", "30M", "60M", "190M", "370M"])
    args = parser.parse_args()

    data = load_data(args.size)
    output_path = OUTPUT_DIR / f"{args.size}_heatmap_grid_paper.pdf"
    make_figure(data, output_path, args.size)


if __name__ == "__main__":
    main()
