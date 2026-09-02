"""
Export *every* Dolma val-loss record we have — the full hyperparameter
sweeps, not just the per-(size, scale, epochs/K) winners that end up in
`dolma_{size}m.py` and therefore in the §6 scaling fit.

Sources (union; the merged/ files miss 100M and 600M, the per-run files
miss two incomplete runs, so we take both):
  results/dolma_val_loss/       — repetition (multi-epoch) runs
  results/dolma_para_val_loss/  — paraphrase runs
  results/dolma_sd_val_loss/    — self-distill runs
  results/hparam/merged/        — same records, aggregated per (scale, size)

Each row is one training run at one (lr, wd).  `selected_for_scaling_fit`
marks the run whose (lr, wd) the corresponding `dolma_*.py` entry
records — i.e. the point that feeds the writeup fits.

Usage:
    python export_hparam_sweeps.py
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import SIZES, TTP_RATIO, load_with_extras  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# individual-run dir → (stream label, merged/ subdir)
STREAMS = [
    ("dolma_val_loss",      "repeat",      "multi_epoch"),
    ("dolma_para_val_loss", "paraphrase",  "para"),
    ("dolma_sd_val_loss",   "selfdistill", "selfdistill"),
]

RUN_RE = re.compile(
    r"^(?P<size>\d+M)_seed(?P<seed>\d+)_"
    r"(?:case(?P<case>\d+)_dolma_epoch(?P<epochs>\d+)"
    r"|dolma_(?P<kind>para|selfdistill)_K(?P<K>\d+))"
    r"_wd(?P<wd>[\d.]+)_lr(?P<lr>[\d.]+e[-+]?\d+)$")


def parse_run(run_name):
    m = RUN_RE.match(run_name)
    if not m:
        return None
    g = m.groupdict()
    return dict(
        size=g["size"].lower(), seed=int(g["seed"]),
        case=int(g["case"]) if g["case"] else None,
        epochs=int(g["epochs"]) if g["epochs"] else 1,
        K=int(g["K"]) if g["K"] else None,
        wd=float(g["wd"]), lr=float(g["lr"]))


def selected_keys():
    """(stream, size, scale, epochs|K, lr, wd) tuples used by dolma_*.py."""
    sel = set()
    for size in SIZES:
        N, datasets, parap, sd = load_with_extras(size)
        for ds in datasets:
            for i, ep in enumerate(ds["epochs"]):
                sel.add(("repeat", size, float(ds["chinchilla_scale"][i]),
                         int(ep), float(ds["learning_rate"][i]),
                         float(ds["weight_decay"][i])))
        for label, group in (("paraphrase", parap), ("selfdistill", sd)):
            for ds in group or []:
                for i, K in enumerate(ds["K"]):
                    sel.add((label, size, float(ds["chinchilla_scale"][i]),
                             int(K), float(ds["learning_rate"][i]),
                             float(ds["weight_decay"][i])))
    return sel


def collect():
    """Union of per-run JSONs and merged/ aggregates, keyed by identity."""
    recs = {}

    def add(stream, scale, size_dir, run_name, payload, origin):
        p = parse_run(run_name)
        if p is None:
            print(f"[warn] unparsed run name: {run_name}")
            return
        key = (stream, scale, run_name)
        if key in recs:                      # per-run file wins over merged
            return
        recs[key] = dict(parsed=p, payload=payload, scale=scale,
                         stream=stream, run_name=run_name, origin=origin,
                         size_dir=size_dir.lower())

    for d, stream, _sub in STREAMS:
        for f in sorted(glob.glob(os.path.join(RESULTS_DIR, d, "*", "*", "*.json"))):
            parts = f.split(os.sep)
            scale = float(parts[-3].replace("chinchilla_", ""))
            for run_name, payload in json.load(open(f)).items():
                add(stream, scale, parts[-2], run_name, payload,
                    os.path.relpath(f, RESULTS_DIR))

    for _d, stream, sub in STREAMS:
        for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "hparam", "merged",
                                                sub, "*.json"))):
            base = os.path.basename(f)[:-5]          # chinchilla_0.5_30M
            head, size_dir = base.rsplit("_", 1)
            scale = float(head.replace("chinchilla_", ""))
            for run_name, payload in json.load(open(f)).items():
                add(stream, scale, size_dir, run_name, payload,
                    os.path.relpath(f, RESULTS_DIR))
    return recs


# Designed grid size per stream — a cell with this many (lr, wd) points is
# "full".  Read off the modal grid shape in the data.
FULL_GRID = {"repeat": 19, "paraphrase": 4, "selfdistill": 6}
SIZE_ORDER = ["14m", "30m", "60m", "100m", "190m", "370m", "600m"]


def coverage_report(rows, out=sys.stdout):
    """Print the per-stream size x scale coverage maps (README §Sweep coverage)."""
    def fmt_tokens(x):
        return f"{x / 1e9:.1f}B" if x >= 1e9 else f"{x / 1e6:.0f}M"

    for stream in ("repeat", "paraphrase", "selfdistill"):
        R = [r for r in rows if r["stream"] == stream]
        if not R:
            continue
        full = FULL_GRID[stream]
        cell = collections.Counter()
        for r in R:
            cell[(r["size"], r["chinchilla_scale"],
                  r["epochs"] if r["epochs"] != "" else r["K"])] += 1
        scales = sorted({r["chinchilla_scale"] for r in R})
        level = "epochs" if stream == "repeat" else "K"

        def mark(n):
            return "F" if n >= full else ("." if n == 1 else "p")

        print(f"\n{stream.upper()} — {len(R)} runs, {len(cell)} cells "
              f"(full grid = {full} points)", file=out)
        print("size   " + "".join(f"{s:>9g}" for s in scales), file=out)
        for sz in SIZE_ORDER:
            if not any(k[0] == sz for k in cell):
                continue
            line = f"{sz:<7}"
            for sc in scales:
                lv = sorted(k[2] for k in cell if k[0] == sz and k[1] == sc)
                line += ("".join(mark(cell[(sz, sc, l)]) for l in lv)
                         or "-").rjust(9)
            print(line, file=out)
        tally = collections.Counter(mark(v) for v in cell.values())
        D = [r["D_tokens"] for r in R if r["D_tokens"] != ""]
        Dfull = [r["D_tokens"] for r in R if r["D_tokens"] != ""
                 and cell[(r["size"], r["chinchilla_scale"],
                           r["epochs"] if r["epochs"] != "" else r["K"])] >= full]
        print(f"  {level} levels: "
              f"{sorted({(r['epochs'] if r['epochs'] != '' else r['K']) for r in R})}",
              file=out)
        print(f"  full={tally['F']}  partial={tally['p']}  single={tally['.']}"
              f"   D: {fmt_tokens(min(D))}-{fmt_tokens(max(D))}"
              + (f"   D where full grid: {fmt_tokens(min(Dfull))}-"
                 f"{fmt_tokens(max(Dfull))}" if Dfull else ""), file=out)

        by_cell = collections.defaultdict(list)
        for r in R:
            by_cell[(r["size"], r["chinchilla_scale"],
                     r["epochs"] if r["epochs"] != "" else r["K"])].append(r)
        grids = collections.Counter()
        for k, sub in by_cell.items():
            if len(sub) < full:            # modal shape among *full* cells only
                continue
            grids[(tuple(sorted({r["learning_rate"] for r in sub})),
                   tuple(sorted({r["weight_decay"] for r in sub})))] += 1
        modal = grids.most_common(1)[0][0]
        print(f"  modal full grid: lr={[f'{x:g}' for x in modal[0]]} "
              f"wd={[f'{x:g}' for x in modal[1]]}", file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data_export/hparam_sweeps")
    ap.add_argument("--coverage", action="store_true",
                    help="print the sweep-coverage maps instead of writing files")
    args = ap.parse_args()

    recs = collect()
    sel = selected_keys()
    print(f"[hparam] {len(recs)} runs collected, "
          f"{len(sel)} (stream, size, scale, epochs/K, lr, wd) keys selected "
          f"by dolma_*.py")

    rows = []
    for r in recs.values():
        p, pay = r["parsed"], r["payload"]
        size = p["size"]
        N = SIZES[size][0] if size in SIZES else None
        if size != r["size_dir"]:
            print(f"[warn] size mismatch {size} vs dir {r['size_dir']}")
        D = r["scale"] * TTP_RATIO * N if N else None
        level = p["epochs"] if r["stream"] == "repeat" else p["K"]
        key = (r["stream"], size, r["scale"], int(level), p["lr"], p["wd"])
        rows.append(dict(
            stream=r["stream"],
            run_name=r["run_name"],
            size=size,
            N_params=int(N) if N else "",
            chinchilla_scale=r["scale"],
            D_tokens=int(round(D)) if D else "",
            epochs=p["epochs"] if r["stream"] == "repeat" else "",
            K=p["K"] if p["K"] is not None else "",
            learning_rate=p["lr"],
            weight_decay=p["wd"],
            seed=p["seed"],
            val_loss=pay.get("validation_loss"),
            perplexity=pay.get("perplexity", ""),
            tokens_evaluated=pay.get("tokens_evaluated", ""),
            run_complete=pay.get("completed", True),
            selected_for_scaling_fit=key in sel,
            source_file=r["origin"],
        ))

    order = {"repeat": 0, "paraphrase": 1, "selfdistill": 2}
    rows.sort(key=lambda r: (order[r["stream"]], r["N_params"] or 0,
                             r["chinchilla_scale"],
                             r["epochs"] or r["K"] or 0,
                             r["learning_rate"], r["weight_decay"]))

    if args.coverage:
        coverage_report(rows)
        return

    FIELDS = list(rows[0].keys())
    outdir = os.path.join(SCRIPT_DIR, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    def write_csv(path, subset):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(subset)

    write_csv(os.path.join(outdir, "hparam_sweep_all.csv"), rows)
    print(f"[hparam] wrote hparam_sweep_all.csv  (n={len(rows)})")
    for stream in ("repeat", "paraphrase", "selfdistill"):
        sub = [r for r in rows if r["stream"] == stream]
        write_csv(os.path.join(outdir, f"hparam_sweep_{stream}.csv"), sub)
        print(f"[hparam] wrote hparam_sweep_{stream}.csv  (n={len(sub)}, "
              f"selected={sum(r['selected_for_scaling_fit'] for r in sub)})")

    n_sel = sum(r["selected_for_scaling_fit"] for r in rows)
    meta = {
        "description": "Every Dolma validation-loss record from the hyperparameter "
                       "sweeps behind the scaling-law fits — all (lr, wd) settings, "
                       "not just the winners.",
        "generated_by": "results/chinchilla_fit_dolma/export_hparam_sweeps.py",
        "n_total": len(rows),
        "n_by_stream": {s: sum(r["stream"] == s for r in rows)
                        for s in ("repeat", "paraphrase", "selfdistill")},
        "n_selected_for_scaling_fit": n_sel,
        "sources": [d for d, _s, _m in STREAMS] + ["hparam/merged"],
        "grid": {
            "learning_rate": sorted({r["learning_rate"] for r in rows}),
            "weight_decay": sorted({r["weight_decay"] for r in rows}),
            "chinchilla_scale": sorted({r["chinchilla_scale"] for r in rows}),
            "sizes": sorted({r["size"] for r in rows},
                            key=lambda s: SIZES[s][0] if s in SIZES else 0),
            "epochs": sorted({r["epochs"] for r in rows if r["epochs"] != ""}),
            "K": sorted({r["K"] for r in rows if r["K"] != ""}),
        },
        "caveats": {
            "tokens_evaluated": "Two validation-set sizes are present "
                                "(3,478,624 and 13,279,917 tokens); losses are "
                                "comparable within a set but the column is kept "
                                "so mixed comparisons stay visible.",
            "run_complete": "False for runs that did not reach their final step; "
                            "their val_loss is the best eval seen so far.",
            "selected_for_scaling_fit": "True when this run's (lr, wd) matches what "
                                        "dolma_<size>.py records for that "
                                        "(scale, epochs/K) — i.e. the point used in "
                                        "the writeup fits.",
        },
        "columns": FIELDS,
    }
    with open(os.path.join(outdir, "hparam_sweep_all.json"), "w") as f:
        json.dump({"metadata": meta, "data": rows}, f, indent=1)
    print(f"[hparam] wrote hparam_sweep_all.json")
    print(f"[hparam] selected_for_scaling_fit = {n_sel} rows")


if __name__ == "__main__":
    main()
