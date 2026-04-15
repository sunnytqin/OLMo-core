"""
Validation BPB vs downstream benchmark performance.
Combines 30M and 370M models on the same plots.

Outputs:
  figures/bpb_vs_benchmarks_paper.pdf  — paper-ready 3-panel horizontal figure
  figures/bpb_vs_<bench>.pdf           — exploratory single-benchmark figures
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_bpb import ce_to_bpb, select_best_runs

# ── paths ────────────────────────────────────────────────────────────────────
EVALS_DIR = Path(__file__).parent / "results"

# ── benchmark metric to extract from results JSON ────────────────────────────
BENCHMARK_METRICS = {
    "hellaswag":        ("acc_norm,none",            "acc_norm",    True),
    "lambada_openai":   ("acc,none",                 "acc",         True),
    "openbookqa":       ("acc_norm,none",            "acc_norm",    True),
    "race":             ("acc,none",                 "acc",         True),
    "gsm8k":            ("exact_match,strict-match", "exact_match", True),
    "gsm8k_bpb_task":   ("bits_per_byte,none",       "bits_per_byte", False),
    "webqs":            ("exact_match,none",         "exact_match", True),
    "squad_completion": ("contains,none",            "contains",    True),
    "c4":               ("bits_per_byte,none",       "bits_per_byte", False),
    "wikitext":         ("bits_per_byte,none",       "bits_per_byte", False),
}

PAPER_BENCHMARKS = ["hellaswag", "lambada_openai", "squad_completion"]
PAPER_BENCH_TITLES = {
    "hellaswag":        "HellaSwag",
    "lambada_openai":   "LAMBADA",
    "squad_completion": "SQuAD (completion)",
}

# ── style ────────────────────────────────────────────────────────────────────
FONT_LABEL  = 18
FONT_TICK   = 16
FONT_LEGEND = 11

cmap_flops = plt.cm.YlOrRd

# Marker: chinchilla scale determines shape
chin_marker = {0.05: "o", 0.1: "s", 0.25: "D"}
chin_label  = {
    0.05: r"0.05× Chin.",
    0.1:  r"0.1× Chin.",
    0.25: r"0.25× Chin.",
}

# Model size: determines fill style (filled=370M, open=30M)
SIZE_STYLE = {
    "370M": {"facecolor": None, "linewidths": 0.5},   # filled (color from flops)
    "30M":  {"facecolor": "none", "linewidths": 1.5},  # open/hollow
}
SIZE_LABEL = {
    "370M": "370M (filled)",
    "30M":  "30M (open)",
}

MODEL_SIZES = ["370M", "30M"]

# ── helpers ───────────────────────────────────────────────────────────────────

def epoch_from_folder(name: str) -> int | None:
    m = re.search(r"epoch(\d+)", name)
    return int(m.group(1)) if m else None


def load_results(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)["results"]


def collect_data(chin_dir: Path, best_runs: list[dict]) -> list[dict]:
    """Collect eval results matched against best runs for a chinchilla scale."""
    best_by_epoch = {r["epoch"]: r for r in best_runs}
    rows = []
    for exp_dir in sorted(chin_dir.iterdir()):
        epoch = epoch_from_folder(exp_dir.name)
        if epoch is None:
            continue
        best = best_by_epoch.get(epoch)
        if best is None:
            continue
        json_files = list(exp_dir.rglob("results_*.json"))
        if not json_files:
            continue
        results = load_results(sorted(json_files)[-1])
        row = {
            "epoch": epoch,
            "val_loss": best["validation_loss"],
            "val_bpb": best["bpb"],
            "flops_multiplier": best["flops_multiplier"],
            "size": best["size"],
        }
        for bench, (key, _, _) in BENCHMARK_METRICS.items():
            row[bench] = results.get(bench, {}).get(key)
        rows.append(row)
    return rows


# ── collect rows for all model sizes and chinchilla scales ───────────────────
# Key: (model_size, chinchilla_scale) -> list of row dicts
all_rows: dict[tuple[str, float], list[dict]] = {}

for model_size in MODEL_SIZES:
    all_best = select_best_runs(size_filter=model_size)
    for chin_scale in sorted(set(r["chinchilla_scale"] for r in all_best)):
        if chin_scale not in chin_marker:
            continue
        chin_dir = EVALS_DIR / f"{model_size}_chinchilla{chin_scale:g}"
        if not chin_dir.exists():
            continue
        runs_for_chin = [r for r in all_best if r["chinchilla_scale"] == chin_scale]
        rows = collect_data(chin_dir, runs_for_chin)
        if rows:
            all_rows[(model_size, chin_scale)] = rows

if not all_rows:
    print("No data found. Run evals first.")
    sys.exit(1)

# ── shared flops colormap ────────────────────────────────────────────────────
all_flops = [r["flops_multiplier"] for rows in all_rows.values()
             for r in rows if r["flops_multiplier"] is not None]
flops_min, flops_max = min(all_flops), max(all_flops)
norm_flops = plt.Normalize(vmin=np.log2(flops_min), vmax=np.log2(flops_max))

FLOPS_TICKS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 16, 32, 64]


def flops_color(f):
    return cmap_flops(norm_flops(np.log2(f)))


def add_colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(cmap=cmap_flops, norm=norm_flops)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.ax.invert_yaxis()
    cbar.set_label("FLOPs (Chinchilla Optimal = 1×)", fontsize=FONT_LEGEND)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    valid_ticks = [v for v in FLOPS_TICKS if flops_min <= v <= flops_max]
    cbar.set_ticks([np.log2(v) for v in valid_ticks])
    cbar.set_ticklabels([f"{v:g}" for v in valid_ticks])
    return cbar


def scatter_point(ax, x, y, chin_scale, model_size, flops_mult, s=80):
    """Plot a single point with marker=chin_scale, fill=model_size, color=flops."""
    c = flops_color(flops_mult)
    style = SIZE_STYLE[model_size]
    if style["facecolor"] == "none":
        ax.scatter(x, y, marker=chin_marker[chin_scale], facecolors="none",
                   edgecolors=c, s=s, zorder=3, linewidths=style["linewidths"])
    else:
        ax.scatter(x, y, marker=chin_marker[chin_scale], color=c,
                   s=s, zorder=3, edgecolors="k", linewidths=style["linewidths"])


# ── output dir ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── paper figure: 1 row × 3 panels ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

all_bpb = [r["val_bpb"] for rows in all_rows.values() for r in rows]
x_min, x_max = min(all_bpb), max(all_bpb)
x_margin = (x_max - x_min) * 0.05
X_LIM = (x_max + x_margin, x_min - x_margin)

for ax, bench in zip(axes, PAPER_BENCHMARKS):
    _, ylabel, is_pct = BENCHMARK_METRICS[bench]
    scale_factor = 100 if is_pct else 1

    for (model_size, chin_scale), rows in all_rows.items():
        valid = [r for r in rows if r.get(bench) is not None
                 and r["flops_multiplier"] is not None]
        for r in valid:
            scatter_point(ax, r["val_bpb"], r[bench] * scale_factor,
                          chin_scale, model_size, r["flops_multiplier"])

    ax.set_xlim(X_LIM)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
    ax.set_title(PAPER_BENCH_TITLES[bench], fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
fig.supxlabel("Validation BPB", fontsize=FONT_LABEL, y=0.04)

add_colorbar(fig, axes[-1])

# Legend: marker shape = chinchilla scale, fill = model size
shape_handles = [
    plt.scatter([], [], marker=chin_marker[s], color="grey",
                s=60, edgecolors="k", linewidths=0.5, label=chin_label[s])
    for s in sorted(s for s in chin_marker
                    if any(s == cs for (_, cs) in all_rows.keys()))
]
size_handles = []
for sz in MODEL_SIZES:
    if any(sz == ms for (ms, _) in all_rows.keys()):
        if SIZE_STYLE[sz]["facecolor"] == "none":
            size_handles.append(
                plt.scatter([], [], marker="o", facecolors="none",
                            edgecolors="grey", s=60, linewidths=1.5,
                            label=SIZE_LABEL[sz]))
        else:
            size_handles.append(
                plt.scatter([], [], marker="o", color="grey",
                            edgecolors="k", s=60, linewidths=0.5,
                            label=SIZE_LABEL[sz]))

axes[0].legend(handles=shape_handles + size_handles, fontsize=FONT_LEGEND,
               title="Data scale / Model size", title_fontsize=FONT_LEGEND,
               loc="best")

fig.tight_layout()
paper_path = OUT_DIR / "bpb_vs_benchmarks_paper.pdf"
fig.savefig(paper_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {paper_path}")

# ── exploratory: one figure per benchmark ────────────────────────────────────
for bench, (_, ylabel, is_pct) in BENCHMARK_METRICS.items():
    scale_factor = 100 if is_pct else 1
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for (model_size, chin_scale), rows in all_rows.items():
        valid = [r for r in rows if r.get(bench) is not None
                 and r["flops_multiplier"] is not None]
        for r in valid:
            scatter_point(ax, r["val_bpb"], r[bench] * scale_factor,
                          chin_scale, model_size, r["flops_multiplier"], s=70)

    ax.invert_xaxis()
    ax.set_xlabel("Validation BPB", fontsize=FONT_LABEL)
    ax.set_ylabel(f"{'Accuracy (%)' if is_pct else ylabel}", fontsize=FONT_LABEL)
    ax.set_title(bench, fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)

    shape_handles = [
        plt.scatter([], [], marker=chin_marker[s], color="grey",
                    s=60, edgecolors="k", linewidths=0.5, label=chin_label[s])
        for s in sorted(s for s in chin_marker
                        if any(s == cs for (_, cs) in all_rows.keys()))
    ]
    size_handles = []
    for sz in MODEL_SIZES:
        if any(sz == ms for (ms, _) in all_rows.keys()):
            if SIZE_STYLE[sz]["facecolor"] == "none":
                size_handles.append(
                    plt.scatter([], [], marker="o", facecolors="none",
                                edgecolors="grey", s=60, linewidths=1.5,
                                label=SIZE_LABEL[sz]))
            else:
                size_handles.append(
                    plt.scatter([], [], marker="o", color="grey",
                                edgecolors="k", s=60, linewidths=0.5,
                                label=SIZE_LABEL[sz]))

    ax.legend(handles=shape_handles + size_handles, fontsize=FONT_LEGEND,
              title="Data scale / Model size", title_fontsize=FONT_LEGEND)
    add_colorbar(fig, ax)

    fig.tight_layout()
    out_path = OUT_DIR / f"bpb_vs_{bench}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
