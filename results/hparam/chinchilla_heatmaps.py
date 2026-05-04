#!/usr/bin/env python3
"""
Generate combined heatmap figures: one per model size.
Rows = chinchilla multiplier (fresh data size), columns = the setting's
"third axis" (epoch for multi_epoch, paraphrase factor K for para).
Each subplot has its own colorbar for local optimal visibility.

Usage:
    python chinchilla_heatmaps.py --setting multi_epoch
    python chinchilla_heatmaps.py --setting para
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MODEL_SIZES = ["14M", "30M", "60M", "190M", "370M"]

# Per-setting config. Each setting differs in:
#   - run name regex / third-axis variable name (epoch vs K)
#   - hparam grid axes
#   - default input/output directories
SETTINGS = {
    "multi_epoch": {
        "run_pattern": re.compile(
            r'(14M|30M|60M|190M|370M)_seed\d+_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)'
        ),
        "third_axis_label": "Epoch",
        "third_axis_short": "ep",
        "weight_decays": [0.1, 0.2, 0.4, 0.8, 1.6],
        "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3],
        "merged_subdir": "merged/multi_epoch",
        "output_subdir": "heatmaps/multi_epoch",
    },
    "para": {
        "run_pattern": re.compile(
            r'(14M|30M|60M|190M|370M)_seed\d+_dolma_para_K(\d+)_wd([\d.]+)_lr([\d.e-]+)'
        ),
        "third_axis_label": "K (paraphrase factor)",
        "third_axis_short": "K",
        "weight_decays": [0.1, 0.2],
        "learning_rates": [1e-3, 3e-3],
        "merged_subdir": "merged/para",
        "output_subdir": "heatmaps/para",
    },
    "selfdistill": {
        "run_pattern": re.compile(
            r'(14M|30M|60M|190M|370M)_seed\d+_dolma_selfdistill_K(\d+)_wd([\d.]+)_lr([\d.e-]+)'
        ),
        "third_axis_label": "K (self-distill factor)",
        "third_axis_short": "K",
        "weight_decays": [0.1, 0.2, 0.4],
        "learning_rates": [1e-3, 3e-3],
        "merged_subdir": "merged/selfdistill",
        "output_subdir": "heatmaps/selfdistill",
    },
}


def parse_run_name(run_name: str, pattern: re.Pattern):
    m = pattern.search(run_name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))


def parse_merged_filename(stem: str):
    # Match longer suffixes first so "190M" isn't caught by "30M"/"90M".
    for size in sorted(MODEL_SIZES, key=len, reverse=True):
        suffix = f"_{size}"
        if stem.endswith(suffix):
            chin = stem[:-len(suffix)]
            return chin, size
    return None, None


def chin_scale(chin_dir: str) -> float:
    return float(chin_dir.replace("chinchilla_", ""))


def load_all_data(merged_dir: Path, pattern: re.Pattern):
    """Returns dict of (chin_dir, model_size, third_axis) -> {(wd, lr): val_loss}"""
    data = {}
    for json_file in sorted(merged_dir.glob("*.json")):
        chin, size = parse_merged_filename(json_file.stem)
        if chin is None:
            continue
        with open(json_file) as f:
            results = json.load(f)
        third_axis_data = defaultdict(dict)
        for run_name, metrics in results.items():
            if "error" in metrics:
                continue
            parsed = parse_run_name(run_name, pattern)
            if parsed is None:
                continue
            _, third, wd, lr = parsed
            val_loss = metrics.get("validation_loss")
            if val_loss is not None:
                third_axis_data[third][(wd, lr)] = val_loss
        for third, grid_data in third_axis_data.items():
            data[(chin, size, third)] = grid_data
    return data


def generate_combined_figure(data, model_size: str, output_path: str, cfg: dict):
    weight_decays = cfg["weight_decays"]
    learning_rates = cfg["learning_rates"]
    short = cfg["third_axis_short"]
    long_label = cfg["third_axis_label"]
    n_wd = len(weight_decays)
    n_lr = len(learning_rates)

    chin_dirs = sorted(
        set(chin for chin, size, _ in data if size == model_size),
        key=chin_scale,
    )
    third_vals = sorted(set(t for _, size, t in data if size == model_size))

    if not chin_dirs or not third_vals:
        print(f"  No data for {model_size}, skipping")
        return

    n_rows = len(chin_dirs)
    n_cols = len(third_vals)

    cell_w = 2.8 if model_size in ("14M", "30M") else 3.2
    cell_h = 2.2 if model_size in ("14M", "30M") else 2.6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w + 1.5, n_rows * cell_h + 1.5),
        squeeze=False,
    )

    for r, chin in enumerate(chin_dirs):
        for c, third in enumerate(third_vals):
            ax = axes[r][c]
            key = (chin, model_size, third)
            chin_label = chin.replace("chinchilla_", "")

            if key not in data or not data[key]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=9, color="#888888", transform=ax.transAxes)
                ax.set_title(f'{chin_label}x / {short}{third}', fontsize=8, fontweight='bold')
                continue

            grid_data = data[key]
            loss_grid = np.full((n_wd, n_lr), np.nan)
            for i, wd in enumerate(weight_decays):
                for j, lr in enumerate(learning_rates):
                    if (wd, lr) in grid_data:
                        loss_grid[i, j] = grid_data[(wd, lr)]

            mask = np.isnan(loss_grid)
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

            best_idx = np.unravel_index(np.nanargmin(loss_grid), loss_grid.shape)
            ax.add_patch(plt.Rectangle(
                (best_idx[1], best_idx[0]), 1, 1,
                fill=False, edgecolor='red', linewidth=2.5,
            ))

            ax.set_title(f'{chin_label}x / {short}{third}', fontsize=8, fontweight='bold')

            if c == 0:
                ax.set_ylabel('WD', fontsize=7)
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

    for r, chin in enumerate(chin_dirs):
        chin_label = chin.replace("chinchilla_", "")
        axes[r][0].annotate(
            f'{chin_label}x',
            xy=(-0.4, 0.5), xycoords='axes fraction',
            fontsize=10, fontweight='bold', ha='center', va='center',
            rotation=90,
        )

    for c, third in enumerate(third_vals):
        axes[0][c].annotate(
            f'{long_label} {third}',
            xy=(0.5, 1.25), xycoords='axes fraction',
            fontsize=9, fontweight='bold', ha='center', va='bottom',
        )

    fig.suptitle(
        f'{model_size} — Validation Loss (WD vs LR)\n'
        f'Rows: Chinchilla multiplier (fresh data size)  |  Columns: {long_label}\n'
        f'Red box = best hyperparams per setting  |  Gray = n/a',
        fontsize=12, fontweight='bold', y=1.02,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate combined heatmap figures")
    parser.add_argument("--setting", choices=list(SETTINGS), required=True,
                        help="Which experiment setting to plot.")
    parser.add_argument("--merged-dir", type=str, default=None,
                        help="Override merged dir (default: merged/<setting>)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output dir (default: heatmaps/<setting>)")
    args = parser.parse_args()

    cfg = SETTINGS[args.setting]
    merged_path = Path(args.merged_dir or cfg["merged_subdir"])
    output_path = Path(args.output_dir or cfg["output_subdir"])

    print(f"[setting={args.setting}] Loading from {merged_path}, writing to {output_path}")
    data = load_all_data(merged_path, cfg["run_pattern"])
    print(f"Found {len(data)} (chinchilla, size, third_axis) combinations")

    for model_size in MODEL_SIZES:
        print(f"\nGenerating {model_size} figure...")
        out_file = output_path / f"{model_size}_heatmap_grid.pdf"
        generate_combined_figure(data, model_size, str(out_file), cfg)

    print("\nDone!")


if __name__ == "__main__":
    main()
