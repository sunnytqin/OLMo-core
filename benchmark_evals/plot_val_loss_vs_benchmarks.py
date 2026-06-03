"""
Validation BPB vs downstream benchmark performance.

Reads the manifest files (manifest_10pct.json, manifest_10pct_para.json) for the
list of evaluated checkpoints + their dolma validation loss, then finds the
corresponding lm_eval results under results/<manifest_id>/.

Produces:
  figures/val_loss_vs_benchmarks.pdf  — one panel per benchmark (4x5 grid)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

# Order by category for the grid layout
PANEL_ORDER = [
    # row 1 — perplexity / BPB on natural text
    "c4", "wikitext",
    # row 1 cont — accuracy tasks
    "lambada_openai", "hellaswag", "openbookqa",
    # row 2 — accuracy/exact-match
    "race", "squad_completion-1", "webqs", "gsm8k",
    # row 2 cont — code BPB
    "humaneval_lm",
    # row 3 — math BPB
    "asdiv_lm", "gsm8k_lm", "mbpp_lm",
    # row 3 cont — QA BPB
    "nq_open_lm", "triviaqa_lm",
    # row 4 — QA BPB cont + instruct BPB
    "webqs_lm", "squad_completion_lm", "ifeval_lm",
]

# Model size -> colormap index (smaller = lighter, larger = darker).
# Each source gets its own colormap so we can compare loss-to-downstream
# between regimes while keeping model size visible.
SIZE_ORDER = ["14M", "30M", "60M", "100M", "190M", "370M", "600M"]
# Use the 0.25–0.95 range so the lightest end is still readable
_LO, _HI = 0.25, 0.95
SOURCE_CMAP = {"multiepoch": plt.cm.Blues, "para": plt.cm.Reds}
SOURCE_LABEL = {"multiepoch": "multi-epoch (D × n)", "para": "paraphrase (D + D'×K)"}


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


def plot_grid(rows, out_path: Path):
    n_panels = len(PANEL_ORDER)
    ncols = 5
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, bench in zip(axes, PANEL_ORDER):
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
        ax.set_title(bench, fontsize=11)
        ax.set_xlabel("Val BPB", fontsize=9)
        ax.set_ylabel(f"{ylabel}{' (%)' if scale == 100 else ''}", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
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
                           linestyle="", markersize=8,
                           markeredgecolor="k", markeredgewidth=0.4,
                           label=f"N={sz}")
            )
    fig.legend(
        handles=handles,
        loc="lower center", ncol=len(SIZE_ORDER) + 1, fontsize=9,
        bbox_to_anchor=(0.5, -0.03), frameon=False,
    )

    fig.suptitle("Validation BPB vs downstream benchmark performance "
                 "(left → right = better model)", fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
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

    plot_grid(rows, OUT_DIR / "val_loss_vs_benchmarks.pdf")
    plot_grid(rows, OUT_DIR / "val_loss_vs_benchmarks.png")


if __name__ == "__main__":
    main()
