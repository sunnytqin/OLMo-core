#!/usr/bin/env python3
"""
Build a manifest of ALL complete runs across 30M, 60M, 370M for scaling law fitting.
Checks actual training progress against expected steps to determine completion.
Skips runs that already have eval results.

Three modes:
  --mode multi_epoch (default): runs named <size>_seed42_case4_dolma_epoch<N>_wd<W>_lr<L>
                                eval results live in results/dolma_val_loss
                                manifests written to results/chinchilla_fit_dolma/
  --mode paraphrase           : runs named <size>_seed42_dolma_para_K<N>_wd<W>_lr<L>
                                eval results live in results/dolma_para_val_loss
                                manifests written to results/chinchilla_fit_dolma_para/
  --mode selfdistill          : runs named <size>_seed42_dolma_selfdistill_K<N>_wd<W>_lr<L>
                                eval results live in results/dolma_sd_val_loss
                                manifests written to results/chinchilla_fit_dolma_sd/
"""
import argparse
import json
import os
import re
import glob
from pathlib import Path

CHECKPOINT_BASE = "/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints"
REPO_RESULTS = "/n/home05/sqin/OLMo-core/results"
BATCH_SIZE = 2097152  # tokens per step

CHINCHILLA_DIRS = [
    "chinchilla_0.05", "chinchilla_0.1", "chinchilla_0.25", "chinchilla_0.5",
    "chinchilla_1", "chinchilla_2", "chinchilla_4", "chinchilla_8", "chinchilla_16",
]
MODEL_SIZES = ["14M", "30M", "60M", "100M", "190M", "370M", "600M"]

MODE_CONFIG = {
    "multi_epoch": {
        "run_pattern": re.compile(
            r"(?P<size>\d+M)_seed42_case4_dolma_epoch(?P<axis>\d+)_wd(?P<wd>[\d.]+)_lr(?P<lr>[\de.-]+)"
        ),
        "axis_field": "epoch",
        "results_base": f"{REPO_RESULTS}/dolma_val_loss",
        "manifest_dir": f"{REPO_RESULTS}/chinchilla_fit_dolma",
    },
    "paraphrase": {
        "run_pattern": re.compile(
            r"(?P<size>\d+M)_seed42_dolma_para_K(?P<axis>\d+)_wd(?P<wd>[\d.]+)_lr(?P<lr>[\de.-]+)"
        ),
        "axis_field": "K",
        "results_base": f"{REPO_RESULTS}/dolma_para_val_loss",
        "manifest_dir": f"{REPO_RESULTS}/chinchilla_fit_dolma_para",
    },
    "selfdistill": {
        "run_pattern": re.compile(
            r"(?P<size>\d+M)_seed42_dolma_selfdistill_K(?P<axis>\d+)_wd(?P<wd>[\d.]+)_lr(?P<lr>[\de.-]+)"
        ),
        "axis_field": "K",
        "results_base": f"{REPO_RESULTS}/dolma_sd_val_loss",
        "manifest_dir": f"{REPO_RESULTS}/chinchilla_fit_dolma_sd",
    },
}


def get_max_step_dir(run_path: str):
    """Return (max_step, has_model_weights) for a run directory."""
    steps = []
    for d in os.listdir(run_path):
        if d.startswith("step") and d != "step0":
            try:
                steps.append(int(d[4:]))
            except ValueError:
                pass
    if not steps:
        return 0, False
    max_step = max(steps)
    model_dir = os.path.join(run_path, f"step{max_step}", "model_and_optim")
    has_model = os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0
    return max_step, has_model


def get_expected_steps(run_path: str):
    """Read config to get expected total steps. Returns None if unknown."""
    cfg_path = os.path.join(run_path, "step0", "config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        md = cfg.get("trainer", {}).get("max_duration", {})
        if isinstance(md, dict) and "value" in md and md.get("unit") == "tokens":
            return md["value"] // BATCH_SIZE
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def is_complete(run_path: str) -> tuple:
    """Check if a run is complete. Returns (complete: bool, max_step: int, reason: str)."""
    max_step, has_model = get_max_step_dir(run_path)
    if max_step == 0 or not has_model:
        return False, max_step, "no checkpoint beyond step0"

    expected = get_expected_steps(run_path)
    if expected is not None:
        if max_step / expected >= 0.99:
            return True, max_step, f"step {max_step}/{expected}"
        else:
            return False, max_step, f"step {max_step}/{expected} ({max_step/expected:.0%})"

    # Older runs without token-based max_duration: trust has_model
    return True, max_step, f"step {max_step} (no target, has model)"


def has_eval_result(results_base: str, chin_dir: str, model_size: str, run_name: str) -> bool:
    """Check if eval result already exists."""
    out_file = Path(results_base) / chin_dir / model_size / f"{run_name}.json"
    if not out_file.exists():
        return False
    try:
        with open(out_file) as f:
            data = json.load(f)
        return run_name in data and "error" not in data[run_name]
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(MODE_CONFIG.keys()),
        default="multi_epoch",
        help="multi_epoch: epoch<N> runs → dolma_val_loss; paraphrase: para_K<N> runs → dolma_para_val_loss",
    )
    args = parser.parse_args()

    cfg = MODE_CONFIG[args.mode]
    run_pattern = cfg["run_pattern"]
    axis_field = cfg["axis_field"]
    results_base = cfg["results_base"]
    manifest_dir = cfg["manifest_dir"]

    print(f"Mode: {args.mode} (axis={axis_field}, results→{results_base})")

    all_complete = []
    all_incomplete = []
    already_evaled = 0
    needs_eval = 0

    for chin_dir in CHINCHILLA_DIRS:
        base = os.path.join(CHECKPOINT_BASE, chin_dir)
        if not os.path.isdir(base):
            continue

        for model_size in MODEL_SIZES:
            runs = sorted(glob.glob(os.path.join(base, f"{model_size}_*")))
            for run_path in runs:
                run_name = os.path.basename(run_path)
                m = run_pattern.match(run_name)
                if not m:
                    continue

                complete, max_step, reason = is_complete(run_path)

                entry = {
                    "run_name": run_name,
                    "chinchilla_dir": chin_dir,
                    "model_size": model_size,
                    axis_field: int(m.group("axis")),
                    "wd": m.group("wd"),
                    "lr": m.group("lr"),
                    "checkpoint_path": os.path.join(run_path, f"step{max_step}"),
                    "last_step": max_step,
                    "complete": complete,
                }

                if complete:
                    evaled = has_eval_result(results_base, chin_dir, model_size, run_name)
                    entry["has_eval"] = evaled
                    all_complete.append(entry)
                    if evaled:
                        already_evaled += 1
                    else:
                        needs_eval += 1
                else:
                    entry["reason"] = reason
                    all_incomplete.append(entry)

    # Summary
    from collections import Counter
    by_size = Counter()
    by_size_eval = Counter()
    for e in all_complete:
        by_size[e["model_size"]] += 1
        if e["has_eval"]:
            by_size_eval[e["model_size"]] += 1

    print(f"\n{'='*60}")
    print(f"SCALING LAW MANIFEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total complete runs: {len(all_complete)}")
    print(f"  Already evaluated: {already_evaled}")
    print(f"  Needs evaluation:  {needs_eval}")
    print(f"Incomplete runs:     {len(all_incomplete)}")
    print()
    for size in MODEL_SIZES:
        print(f"  {size}: {by_size[size]} complete ({by_size_eval[size]} already evaled)")
    print()

    # Print incomplete summary
    if all_incomplete:
        print("Incomplete runs:")
        for e in all_incomplete:
            print(f"  {e['chinchilla_dir']} {e['run_name']}: {e['reason']}")
        print()

    # Write manifests
    out_dir = Path(manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full manifest (all complete runs)
    manifest = {
        "checkpoints": all_complete,
        "summary": {
            "total_complete": len(all_complete),
            "already_evaluated": already_evaled,
            "needs_evaluation": needs_eval,
            "by_model_size": {size: by_size[size] for size in MODEL_SIZES},
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Full manifest written to {manifest_path}")

    # Eval-only manifest (only runs that need eval)
    eval_entries = [e for e in all_complete if not e["has_eval"]]
    eval_manifest = {
        "checkpoints": eval_entries,
        "summary": {"total": len(eval_entries)},
    }
    eval_manifest_path = out_dir / "manifest_needs_eval.json"
    with open(eval_manifest_path, "w") as f:
        json.dump(eval_manifest, f, indent=2)
    print(f"Eval manifest written to {eval_manifest_path} ({len(eval_entries)} runs)")


if __name__ == "__main__":
    main()
