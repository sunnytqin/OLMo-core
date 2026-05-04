#!/usr/bin/env python3
"""
Heatmap matrices showing optimal LR and WD as a function of fresh data
size (chinchilla multiplier) and the setting's third axis (epoch for
multi_epoch, paraphrase factor K for para).
One figure per model size, two subplots: optimal LR and optimal WD.

Cells are only considered "explored" (and colored) if more than
`MIN_CONFIGS_FOR_OPTIMAL` unique (wd, lr) configs were run for that
(chinchilla, third_axis) setting. Otherwise, the cell is grayed out
with "n/a".

Usage:
    python optimal_hparams_plot.py --setting multi_epoch
    python optimal_hparams_plot.py --setting para
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

MODEL_SIZES = ["14M", "30M", "60M", "190M", "370M"]

# Per-setting config. `min_configs` is the threshold below which we gray
# out a cell and label it "n/a". multi_epoch sweeps 5x4=20, so 5 is a
# meaningful "explored enough" floor. para sweeps 2x2=4, so 4 means the
# full grid was run. selfdistill sweeps 3x2=6, so 6 means the full grid.
SETTINGS = {
    "multi_epoch": {
        "run_pattern": re.compile(
            r'(14M|30M|60M|190M|370M)_seed\d+_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)'
        ),
        "third_axis_label": "Epochs",
        "merged_subdir": "merged/multi_epoch",
        "output_subdir": "heatmaps/multi_epoch",
        "min_configs": 5,
    },
    "para": {
        "run_pattern": re.compile(
            r'(14M|30M|60M|190M|370M)_seed\d+_dolma_para_K(\d+)_wd([\d.]+)_lr([\d.e-]+)'
        ),
        "third_axis_label": "K (paraphrase factor)",
        "merged_subdir": "merged/para",
        "output_subdir": "heatmaps/para",
        "min_configs": 4,
    },
    "selfdistill": {
        "run_pattern": re.compile(
            r'(14M|30M|60M|190M|370M)_seed\d+_dolma_selfdistill_K(\d+)_wd([\d.]+)_lr([\d.e-]+)'
        ),
        "third_axis_label": "K (self-distill factor)",
        "merged_subdir": "merged/selfdistill",
        "output_subdir": "heatmaps/selfdistill",
        "min_configs": 6,
    },
}


def parse_run_name(run_name, pattern):
    m = pattern.search(run_name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))


def parse_merged_filename(stem):
    for size in sorted(MODEL_SIZES, key=len, reverse=True):
        suffix = f"_{size}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)], size
    return None, None


def chin_scale(chin_dir):
    return float(chin_dir.replace("chinchilla_", ""))


def load_optimal_hparams(merged_dir, pattern):
    """For each (chin, size, third_axis), collect the grid of configs and
    record the best plus the count. Threshold is applied at plot time."""
    records = []
    for json_file in sorted(Path(merged_dir).glob("*.json")):
        chin, size = parse_merged_filename(json_file.stem)
        if chin is None:
            continue
        with open(json_file) as f:
            results = json.load(f)
        third_data = defaultdict(dict)
        for run_name, metrics in results.items():
            if "error" in metrics:
                continue
            parsed = parse_run_name(run_name, pattern)
            if parsed is None:
                continue
            _, third, wd, lr = parsed
            val_loss = metrics.get("validation_loss")
            if val_loss is not None:
                third_data[third][(wd, lr)] = val_loss
        for third, grid in third_data.items():
            if not grid:
                continue
            n_configs = len(grid)
            best_key = min(grid, key=grid.get)
            records.append({
                "chin": chin_scale(chin),
                "size": size,
                "third": third,
                "best_lr": best_key[1],
                "best_wd": best_key[0],
                "best_loss": grid[best_key],
                "n_configs": n_configs,
            })
    return records


def plot_model_figure(records, model_size, output_path, third_axis_label, min_configs):
    recs = [r for r in records if r["size"] == model_size]
    if not recs:
        print(f"  No data for {model_size}")
        return

    chins = sorted(set(r["chin"] for r in recs))
    thirds = sorted(set(r["third"] for r in recs))

    lookup = {(r["chin"], r["third"]): r for r in recs}

    explored = [r for r in recs if r["n_configs"] >= min_configs]
    all_lrs = sorted(set(r["best_lr"] for r in explored))
    all_wds = sorted(set(r["best_wd"] for r in explored))

    fig, (ax_lr, ax_wd) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, param, all_vals, label in [
        (ax_lr, "best_lr", all_lrs, "Optimal LR"),
        (ax_wd, "best_wd", all_wds, "Optimal WD"),
    ]:
        n_chins = len(chins)
        n_thirds = len(thirds)

        val_to_idx = {v: i for i, v in enumerate(all_vals)}
        n_vals = len(all_vals)

        grid = np.full((n_thirds, n_chins), np.nan)
        annot = np.full((n_thirds, n_chins), "", dtype=object)
        undersampled = np.zeros((n_thirds, n_chins), dtype=bool)

        for i, third in enumerate(thirds):
            for j, chin in enumerate(chins):
                key = (chin, third)
                if key not in lookup:
                    continue
                rec = lookup[key]
                if rec["n_configs"] < min_configs:
                    undersampled[i, j] = True
                    annot[i, j] = f"n/a\n({rec['n_configs']})"
                    continue
                val = rec[param]
                grid[i, j] = val_to_idx[val]
                if param == "best_lr":
                    annot[i, j] = f"{val:.0e}"
                else:
                    annot[i, j] = f"{val}"

        mask = np.isnan(grid)

        if n_vals > 0:
            cmap = plt.get_cmap("RdYlGn_r", n_vals)
            bounds = np.arange(-0.5, n_vals)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto",
                           interpolation="nearest")
        else:
            im = ax.imshow(np.zeros((n_thirds, n_chins)), cmap="gray",
                           vmin=0, vmax=1, aspect="auto",
                           interpolation="nearest", alpha=0)

        for i in range(n_thirds):
            for j in range(n_chins):
                if mask[i, j]:
                    facecolor = "#b0b0b0" if undersampled[i, j] else "#e0e0e0"
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=True, facecolor=facecolor,
                                               edgecolor="white", linewidth=0.5))

        for i in range(n_thirds):
            for j in range(n_chins):
                if annot[i, j] == "":
                    continue
                if mask[i, j]:
                    ax.text(j, i, annot[i, j], ha="center", va="center",
                            fontsize=8, color="#333333")
                else:
                    color = "white" if grid[i, j] > n_vals / 2 else "black"
                    ax.text(j, i, annot[i, j], ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(n_chins))
        ax.set_xticklabels([f"{c}x" for c in chins], fontsize=9)
        ax.set_yticks(range(n_thirds))
        ax.set_yticklabels([str(t) for t in thirds], fontsize=9)
        ax.set_xlabel("Chinchilla Multiplier (fresh data)", fontweight="bold")
        ax.set_ylabel(third_axis_label, fontweight="bold")
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")

        if n_vals > 0:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_ticks(range(n_vals))
            if param == "best_lr":
                cbar.set_ticklabels([f"{v:.0e}" for v in all_vals])
            else:
                cbar.set_ticklabels([f"{v}" for v in all_vals])

        ax.set_xticks(np.arange(-0.5, n_chins), minor=True)
        ax.set_yticks(np.arange(-0.5, n_thirds), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", size=0)

    fig.suptitle(
        f"{model_size} — Optimal Hyperparameters  "
        f"(gray = <{min_configs} configs swept)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=list(SETTINGS), required=True,
                        help="Which experiment setting to plot.")
    parser.add_argument("--merged-dir", default=None,
                        help="Override merged dir (default: merged/<setting>)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output dir (default: heatmaps/<setting>)")
    args = parser.parse_args()

    cfg = SETTINGS[args.setting]
    merged_dir = args.merged_dir or cfg["merged_subdir"]
    output_dir = args.output_dir or cfg["output_subdir"]

    records = load_optimal_hparams(merged_dir, cfg["run_pattern"])
    print(f"Loaded {len(records)} (chin, size, third_axis) cells")

    for size in MODEL_SIZES:
        print(f"\nGenerating {size} figure...")
        out = str(Path(output_dir) / f"{size}_optimal_hparams.pdf")
        plot_model_figure(records, size, out, cfg["third_axis_label"],
                          cfg["min_configs"])

    print("\nDone!")


if __name__ == "__main__":
    main()
