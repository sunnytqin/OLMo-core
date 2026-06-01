"""
Build a JSON manifest of checkpoints to benchmark-eval.

Two sources, both keyed on (size, chinchilla_scale, <secondary>):
  --source multiepoch  reads results/dolma_val_loss/, secondary key = epoch
  --source para        reads results/dolma_para_val_loss/, secondary key = K (paraphrase seeds)

For each cell, picks the best (lr, wd) by validation_loss, resolves it to the
final-step checkpoint dir on netscratch, then takes a stratified random sample
within each (size, chinchilla_scale) cell.

Usage:
    python build_manifest.py --source multiepoch --frac 0.1 --seed 42 --out manifest_10pct.json
    python build_manifest.py --source para       --frac 0.1 --seed 42 --out manifest_10pct_para.json
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CHECKPOINTS_ROOT = Path("/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints")

SOURCES = {
    "multiepoch": {
        "val_loss_dir": REPO_ROOT / "results" / "dolma_val_loss",
        "regex": re.compile(r'(\d+M)_seed\d+_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)'),
        "secondary": "epoch",
    },
    "para": {
        "val_loss_dir": REPO_ROOT / "results" / "dolma_para_val_loss",
        "regex": re.compile(r'(\d+M)_seed\d+_dolma_para_K(\d+)_wd([\d.]+)_lr([\d.e-]+)'),
        "secondary": "K",
    },
}


def collect_best_runs(source: str):
    """Scan a val-loss source dir, return best (lr, wd) record per cell."""
    cfg = SOURCES[source]
    grid = defaultdict(dict)  # (chin, size, secondary) -> {(wd, lr, run_name): val_loss}

    for chin_dir in sorted(cfg["val_loss_dir"].iterdir()):
        if not chin_dir.is_dir() or not chin_dir.name.startswith("chinchilla_"):
            continue
        chin_scale = float(chin_dir.name.replace("chinchilla_", ""))

        for size_dir in sorted(chin_dir.iterdir()):
            if not size_dir.is_dir():
                continue
            size = size_dir.name

            for json_file in sorted(size_dir.glob("*.json")):
                m = cfg["regex"].search(json_file.stem)
                if not m:
                    continue
                _, secondary, wd, lr = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))

                with open(json_file) as f:
                    data = json.load(f)
                metrics = next(iter(data.values()))
                if "error" in metrics:
                    continue
                val_loss = metrics.get("validation_loss")
                if val_loss is None:
                    continue
                grid[(chin_scale, size, secondary)][(wd, lr, json_file.stem)] = val_loss

    best = []
    for (chin, size, secondary), hparams in grid.items():
        key = min(hparams, key=hparams.get)
        wd, lr, run_name = key
        best.append({
            "size": size,
            "chinchilla_scale": chin,
            cfg["secondary"]: secondary,
            "lr": lr,
            "wd": wd,
            "validation_loss": hparams[key],
            "run_name": run_name,
            "source": source,
        })
    return best


def resolve_checkpoint_path(record: dict):
    """Find the final-step checkpoint dir for a run. Returns (path, step) or (None, None)."""
    run_dir = CHECKPOINTS_ROOT / f"chinchilla_{record['chinchilla_scale']:g}" / record["run_name"]
    if not run_dir.is_dir():
        return None, None
    steps = []
    for entry in run_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("step"):
            continue
        try:
            n = int(entry.name[len("step"):])
        except ValueError:
            continue
        if n == 0:
            continue
        if not (entry / "model_and_optim").is_dir():
            continue
        steps.append((n, entry))
    if not steps:
        return None, None
    n, path = max(steps)
    return path, n


def stratified_sample(records, frac: float, seed: int):
    """Sample `frac` of records within each (size, chinchilla_scale) cell. Always keep >=1."""
    rng = random.Random(seed)
    by_cell = defaultdict(list)
    for r in records:
        by_cell[(r["size"], r["chinchilla_scale"])].append(r)
    sampled = []
    for cell, items in sorted(by_cell.items()):
        k = max(1, round(len(items) * frac))
        sampled.extend(rng.sample(items, min(k, len(items))))
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Build benchmark-eval manifest")
    parser.add_argument("--source", choices=sorted(SOURCES.keys()), default="multiepoch",
                        help="Which val-loss source to read (multiepoch or para)")
    parser.add_argument("--frac", type=float, default=0.1,
                        help="Fraction to sample within each (size, scale) cell")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None,
                        help="Output manifest path (default depends on --source)")
    parser.add_argument("--no-sample", action="store_true",
                        help="Skip sampling — emit every best cell")
    args = parser.parse_args()

    cfg = SOURCES[args.source]
    out_path = Path(args.out) if args.out else REPO_ROOT / "benchmark_evals" / f"manifest_{int(args.frac*100)}pct_{args.source}.json"

    best = collect_best_runs(args.source)
    print(f"[source={args.source}] Collected {len(best)} best-run records across all (size, scale, {cfg['secondary']}) cells")

    resolved = []
    missing = []
    for r in best:
        path, step = resolve_checkpoint_path(r)
        if path is None:
            missing.append(r)
            continue
        r["checkpoint_path"] = str(path)
        r["step"] = step
        r["manifest_id"] = f"{r['run_name']}_step{step}"
        resolved.append(r)
    print(f"Resolved {len(resolved)} to disk paths; {len(missing)} missing checkpoint dirs")

    if args.no_sample:
        chosen = resolved
    else:
        chosen = stratified_sample(resolved, args.frac, args.seed)
        print(f"Sampled {len(chosen)} entries at frac={args.frac}, seed={args.seed}")

    by_size_scale = defaultdict(int)
    for r in chosen:
        by_size_scale[(r["size"], r["chinchilla_scale"])] += 1
    print("\nSampled per (size, chinchilla_scale):")
    for (size, scale), n in sorted(by_size_scale.items()):
        print(f"  {size:>5} chin={scale:<5g}  {n}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(chosen, f, indent=2)
    print(f"\nWrote {len(chosen)} entries -> {out_path}")


if __name__ == "__main__":
    main()
