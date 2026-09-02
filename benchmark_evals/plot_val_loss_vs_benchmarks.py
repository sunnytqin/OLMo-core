"""
Validation BPB vs downstream benchmark performance.

Reads the manifest files (manifest_10pct.json, manifest_10pct_para.json) for the
list of evaluated checkpoints + their dolma validation loss, then finds the
corresponding lm_eval results under results/<manifest_id>/.

Produces:
  figures/val_loss_vs_benchmarks_acc.pdf   — accuracy/EM breakdown grid
  figures/val_loss_vs_benchmarks_bpb.pdf   — BPB breakdown grid
  figures/val_loss_vs_benchmarks_paper.pdf — 2-panel aggregated figure
"""
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# Palatino Linotype — matches isoloss_contour.py styling.
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'results', 'fonts')
for _f in glob.glob(os.path.join(_FONT_DIR, 'palatinolinotype_*.ttf')):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Palatino Linotype', 'P052', 'Palatino', 'serif'],
    'mathtext.fontset': 'cm',
})

# nats/token -> bits/byte. Measured on C4 val with allenai/dolma2-tokenizer.
BYTES_PER_TOKEN = 4.6954


def ce_to_bpb(loss_nats: float) -> float:
    return loss_nats / math.log(2) / BYTES_PER_TOKEN

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

MANIFESTS = [
    ROOT / "manifest_10pct.json",
    ROOT / "manifest_10pct_para.json",
]

# benchmark_name -> (json_metric_key, ylabel, higher_is_better)
BENCHMARK_METRICS = {
    "c4":                  ("bits_per_byte,none",      "BPB",        False),
    "wikitext":            ("bits_per_byte,none",      "BPB",        False),
    "lambada_openai":      ("acc,none",                "Accuracy",   True),
    "hellaswag":           ("acc_norm,none",           "Acc (norm)", True),
    "openbookqa":          ("acc_norm,none",           "Acc (norm)", True),
    "race":                ("acc,none",                "Accuracy",   True),
    "squad_completion-1":  ("contains,none",           "Contains",   True),
    "webqs":               ("exact_match,none",        "EM",         True),
    "gsm8k":               ("exact_match,strict-match", "EM",        True),
    "asdiv_lm":            ("bits_per_byte,none",      "BPB",        False),
    "gsm8k_lm":            ("bits_per_byte,none",      "BPB",        False),
    "humaneval_lm":        ("bits_per_byte,none",      "BPB",        False),
    "mbpp_lm":             ("bits_per_byte,none",      "BPB",        False),
    "nq_open_lm":          ("bits_per_byte,none",      "BPB",        False),
    "triviaqa_lm":         ("bits_per_byte,none",      "BPB",        False),
    "webqs_lm":            ("bits_per_byte,none",      "BPB",        False),
    "squad_completion_lm": ("bits_per_byte,none",      "BPB",        False),
    "ifeval_lm":           ("bits_per_byte,none",      "BPB",        False),
}

ACC_PANEL_ORDER = [
    "lambada_openai", "hellaswag", "openbookqa", "race", "squad_completion-1",
]

BPB_PANEL_ORDER = [
    "c4", "wikitext",
    "gsm8k_lm", "humaneval_lm", "mbpp_lm",
    "nq_open_lm", "triviaqa_lm", "webqs_lm", "squad_completion_lm", "ifeval_lm",
]

BENCH_DISPLAY_NAME = {
    "c4":                  "C4",
    "wikitext":            "WikiText-103",
    "lambada_openai":      "LAMBADA",
    "hellaswag":           "HellaSwag",
    "openbookqa":          "OpenBookQA",
    "race":                "RACE",
    "squad_completion-1":  "SQuAD",
    "gsm8k_lm":            "GSM8K (LM)",
    "humaneval_lm":        "HumanEval (LM)",
    "mbpp_lm":             "MBPP (LM)",
    "nq_open_lm":          "NQ Open (LM)",
    "triviaqa_lm":         "TriviaQA (LM)",
    "webqs_lm":            "WebQS (LM)",
    "squad_completion_lm": "SQuAD (LM)",
    "ifeval_lm":           "IFEval (LM)",
}

# Model size -> colormap index (smaller = lighter, larger = darker).
# Each source gets its own colormap so we can compare loss-to-downstream
# between regimes while keeping model size visible.
SIZE_ORDER = ["14M", "30M", "60M", "100M", "190M", "370M", "600M"]
# Use the 0.25–0.95 range so the lightest end is still readable
_LO, _HI = 0.25, 0.95
SOURCE_CMAP = {"multiepoch": plt.cm.Blues, "para": plt.cm.Reds}
SOURCE_LABEL = {"multiepoch": "multi-epoch", "para": "paraphrase"}


def size_color(size: str, source: str):
    i = SIZE_ORDER.index(size)
    frac = _LO + (_HI - _LO) * (i / (len(SIZE_ORDER) - 1))
    return SOURCE_CMAP[source](frac)


def find_results_json(manifest_id: str) -> Path | None:
    """Find the latest results_*.json under results/<manifest_id>/."""
    run_dir = RESULTS_DIR / manifest_id
    if not run_dir.is_dir():
        return None
    candidates = list(run_dir.rglob("results_*.json"))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def collect_rows():
    """Load each manifest entry's eval results into a flat list of row dicts."""
    rows = []
    for manifest_path in MANIFESTS:
        if not manifest_path.exists():
            continue
        for entry in json.load(open(manifest_path)):
            rj_path = find_results_json(entry["manifest_id"])
            if rj_path is None:
                continue
            results = json.load(open(rj_path))["results"]
            row = {
                "manifest_id": entry["manifest_id"],
                "size": entry["size"],
                "chinchilla_scale": entry["chinchilla_scale"],
                "val_loss": entry["validation_loss"],
                "val_bpb": ce_to_bpb(entry["validation_loss"]),
                "source": entry.get("source", "multiepoch"),
            }
            # secondary key (epoch or K) for hover/debug
            if "epoch" in entry:
                row["secondary"] = ("epoch", entry["epoch"])
            elif "K" in entry:
                row["secondary"] = ("K", entry["K"])
            for bench, (key, _, _) in BENCHMARK_METRICS.items():
                v = results.get(bench, {}).get(key)
                row[bench] = v if isinstance(v, (int, float)) else None
            rows.append(row)
    return rows


def plot_grid(rows, out_path: Path, panel_order, title=None, legend_fontsize=20, legend_y=-0.06):
    n_panels = len(panel_order)
    ncols = 5
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, max(4.0, 3.4 * nrows)))
    axes = np.asarray(axes).reshape(-1)

    for ax, bench in zip(axes, panel_order):
        _, ylabel, higher_is_better = BENCHMARK_METRICS[bench]
        scale = 100 if ylabel.startswith("Acc") or ylabel in ("EM", "Contains") else 1
        for r in rows:
            if r[bench] is None:
                continue
            ax.scatter(
                r["val_bpb"], r[bench] * scale,
                marker="o",
                color=size_color(r["size"], r["source"]),
                s=42,
                edgecolors="k", linewidths=0.4, alpha=0.9, zorder=3,
            )
        ax.set_title(BENCH_DISPLAY_NAME.get(bench, bench), fontsize=20)
        ax.set_xlabel("Val BPB", fontsize=17)
        ax.set_ylabel(f"{ylabel}{' (%)' if scale == 100 else ''}", fontsize=17)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # left -> right means better (lower BPB)

    # Hide unused axes
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Shared legend: two rows of size swatches, one per source colormap.
    handles = []
    for source in ("multiepoch", "para"):
        handles.append(
            plt.Line2D([], [], linestyle="", marker="", label=SOURCE_LABEL[source])
        )
        for sz in SIZE_ORDER:
            handles.append(
                plt.Line2D([], [], marker="o", color=size_color(sz, source),
                           linestyle="", markersize=10,
                           markeredgecolor="k", markeredgewidth=0.4,
                           label=f"N={sz}")
            )
    fig.legend(
        handles=handles,
        loc="lower center", ncol=len(SIZE_ORDER) + 1, fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, legend_y), frameon=False,
    )

    suptitle = title or r"Validation BPB vs downstream benchmark performance (lower BPB $\rightarrow$ better model)"
    fig.suptitle(suptitle, fontsize=18, y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


BPB_TASKS = [
    "c4", "wikitext", "gsm8k_lm", "humaneval_lm", "mbpp_lm",
    "nq_open_lm", "triviaqa_lm", "webqs_lm", "squad_completion_lm", "ifeval_lm",
]
ACC_TASKS = [
    "lambada_openai", "hellaswag", "openbookqa", "race", "squad_completion-1",
]


def plot_paper(rows, out_path: Path):
    """Two-panel paper figure: averaged BPB and averaged accuracy vs val BPB."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))

    panel_cfg = [
        (axes[0], BPB_TASKS,  r"Avg. BPB ($\downarrow$)",  False),
        (axes[1], ACC_TASKS,  r"Avg. Accuracy % ($\uparrow$)", True),
    ]

    for ax, tasks, ylabel, higher_is_better in panel_cfg:
        for r in rows:
            vals = [r[t] for t in tasks if r[t] is not None]
            if not vals:
                continue
            avg = np.mean(vals)
            if higher_is_better:
                avg *= 100
            ax.scatter(
                r["val_bpb"], avg,
                marker="o",
                color=size_color(r["size"], r["source"]),
                s=42,
                edgecolors="k", linewidths=0.4, alpha=0.9, zorder=3,
            )
        ax.set_xlabel(r"Val BPB ($\downarrow$)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    handles = []
    for source in ("multiepoch", "para"):
        handles.append(
            plt.Line2D([], [], linestyle="", marker="", label=SOURCE_LABEL[source])
        )
        for sz in SIZE_ORDER:
            handles.append(
                plt.Line2D([], [], marker="o", color=size_color(sz, source),
                           linestyle="", markersize=7,
                           markeredgecolor="k", markeredgewidth=0.4,
                           label=f"N={sz}")
            )
    fig.legend(
        handles=handles,
        loc="lower center", ncol=len(SIZE_ORDER) + 1, fontsize=10,
        bbox_to_anchor=(0.5, -0.10), frameon=False,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1.0))
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    rows = collect_rows()
    if not rows:
        print("No data found — make sure results/ has entries matching the manifests.")
        return
    by_source = defaultdict(int)
    by_size = defaultdict(int)
    for r in rows:
        by_source[r["source"]] += 1
        by_size[r["size"]] += 1
    print(f"Loaded {len(rows)} rows")
    print(f"  by source: {dict(by_source)}")
    print(f"  by size:   {dict(by_size)}")

    _acc_title = r"Val BPB vs Accuracy / EM Benchmarks" + "\n" + r"(Acc $\uparrow$, BPB $\downarrow$)"
    _bpb_title = r"Val BPB vs BPB Benchmarks" + "\n" + r"(BPB $\downarrow$)"
    plot_grid(rows, OUT_DIR / "val_loss_vs_benchmarks_acc.pdf",
              panel_order=ACC_PANEL_ORDER, title=_acc_title, legend_y=-0.12)
    plot_grid(rows, OUT_DIR / "val_loss_vs_benchmarks_acc.png",
              panel_order=ACC_PANEL_ORDER, title=_acc_title, legend_y=-0.12)
    plot_grid(rows, OUT_DIR / "val_loss_vs_benchmarks_bpb.pdf",
              panel_order=BPB_PANEL_ORDER, title=_bpb_title, legend_fontsize=20)
    plot_grid(rows, OUT_DIR / "val_loss_vs_benchmarks_bpb.png",
              panel_order=BPB_PANEL_ORDER, title=_bpb_title, legend_fontsize=20)
    plot_paper(rows, OUT_DIR / "val_loss_vs_benchmarks_paper.pdf")
    plot_paper(rows, OUT_DIR / "val_loss_vs_benchmarks_paper.png")


if __name__ == "__main__":
    main()
