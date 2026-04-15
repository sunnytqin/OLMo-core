#!/usr/bin/env python3
"""
Heatmap matrices showing optimal LR and WD as a function of
fresh data size (chinchilla multiplier) and epochs.
One figure per model size, two subplots: optimal LR and optimal WD.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

RUN_PATTERN = re.compile(
    r'(30M|60M|190M|370M)_seed\d+_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)'
)


def parse_run_name(run_name):
    m = RUN_PATTERN.search(run_name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))


def parse_merged_filename(stem):
    for size in ["370M", "190M", "60M", "30M"]:
        suffix = f"_{size}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)], size
    return None, None


def chin_scale(chin_dir):
    return float(chin_dir.replace("chinchilla_", ""))


def load_optimal_hparams(merged_dir):
    records = []
    for json_file in sorted(Path(merged_dir).glob("*.json")):
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
        for epoch, grid in epoch_data.items():
            if not grid:
                continue
            best_key = min(grid, key=grid.get)
            records.append({
                "chin": chin_scale(chin),
                "size": size,
                "epoch": epoch,
                "best_lr": best_key[1],
                "best_wd": best_key[0],
                "best_loss": grid[best_key],
            })
    return records


def plot_model_figure(records, model_size, output_path):
    recs = [r for r in records if r["size"] == model_size]
    if not recs:
        print(f"  No data for {model_size}")
        return

    chins = sorted(set(r["chin"] for r in recs))
    epochs = sorted(set(r["epoch"] for r in recs))

    # Build lookup
    lookup = {}
    for r in recs:
        lookup[(r["chin"], r["epoch"])] = r

    # All possible LR and WD values (for discrete colormap)
    all_lrs = sorted(set(r["best_lr"] for r in recs))
    all_wds = sorted(set(r["best_wd"] for r in recs))

    fig, (ax_lr, ax_wd) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, param, all_vals, label in [
        (ax_lr, "best_lr", all_lrs, "Optimal LR"),
        (ax_wd, "best_wd", all_wds, "Optimal WD"),
    ]:
        n_chins = len(chins)
        n_epochs = len(epochs)

        # Map param values to indices for discrete colormap
        val_to_idx = {v: i for i, v in enumerate(all_vals)}
        n_vals = len(all_vals)

        grid = np.full((n_epochs, n_chins), np.nan)
        annot = np.full((n_epochs, n_chins), "", dtype=object)

        for i, epoch in enumerate(epochs):
            for j, chin in enumerate(chins):
                key = (chin, epoch)
                if key in lookup:
                    val = lookup[key][param]
                    grid[i, j] = val_to_idx[val]
                    if param == "best_lr":
                        annot[i, j] = f"{val:.0e}"
                    else:
                        annot[i, j] = f"{val}"

        mask = np.isnan(grid)

        # Discrete colormap
        cmap = plt.cm.get_cmap("RdYlGn_r", n_vals)
        bounds = np.arange(-0.5, n_vals)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest")

        # Annotate
        for i in range(n_epochs):
            for j in range(n_chins):
                if not mask[i, j]:
                    ax.text(j, i, annot[i, j], ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if grid[i, j] > n_vals / 2 else "black")

        # Gray out missing cells
        for i in range(n_epochs):
            for j in range(n_chins):
                if mask[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=True, facecolor="#e0e0e0",
                                               edgecolor="white", linewidth=0.5))

        ax.set_xticks(range(n_chins))
        ax.set_xticklabels([f"{c}x" for c in chins], fontsize=9)
        ax.set_yticks(range(n_epochs))
        ax.set_yticklabels([str(e) for e in epochs], fontsize=9)
        ax.set_xlabel("Chinchilla Multiplier (fresh data)", fontweight="bold")
        ax.set_ylabel("Epochs", fontweight="bold")
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")

        # Colorbar with actual values
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks(range(n_vals))
        if param == "best_lr":
            cbar.set_ticklabels([f"{v:.0e}" for v in all_vals])
        else:
            cbar.set_ticklabels([f"{v}" for v in all_vals])

        # Grid lines
        ax.set_xticks(np.arange(-0.5, n_chins), minor=True)
        ax.set_yticks(np.arange(-0.5, n_epochs), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", size=0)

    fig.suptitle(f"{model_size} — Optimal Hyperparameters",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", default="merged")
    parser.add_argument("--output-dir", default="heatmaps")
    args = parser.parse_args()

    records = load_optimal_hparams(args.merged_dir)
    print(f"Loaded {len(records)} optimal points")

    for size in ["30M", "60M", "190M", "370M"]:
        print(f"\nGenerating {size} figure...")
        out = str(Path(args.output_dir) / f"{size}_optimal_hparams.pdf")
        plot_model_figure(records, size, out)

    print("\nDone!")


if __name__ == "__main__":
    main()
