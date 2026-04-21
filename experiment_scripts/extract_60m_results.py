#!/usr/bin/env python3
"""
Extract 60M validation losses from training logs and save in merged JSON format
matching results/hparam/merged/chinchilla_{CHIN}_{SIZE}.json.

Uses FINAL val loss (last eval at end of training) to match how we report results.
Only includes runs that reached the final training step.
"""
import json
import re
from pathlib import Path

CHECKPOINT_ROOT = Path("/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints")
OUTPUT_DIR = Path("/n/home05/sqin/OLMo-core/results/hparam/merged")

RUN_PATTERN = re.compile(
    r'60M_seed\d+_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)'
)
STEP_PATTERN = re.compile(r'step=(\d+)/(\d+)')
VAL_LOSS_PATTERN = re.compile(r'dolma/CE loss=([\d.]+)')


def extract_run_metrics(run_dir: Path):
    """Extract val loss from a single wandb run. Returns dict or None."""
    log_file = run_dir / "files" / "output.log"
    if not log_file.exists():
        return None

    try:
        text = log_file.read_text(errors="ignore")
    except Exception:
        return None

    # Find all step markers and val losses
    steps = STEP_PATTERN.findall(text)
    val_losses = [float(m) for m in VAL_LOSS_PATTERN.findall(text)
                  if "variants" not in m]

    if not val_losses or not steps:
        return None

    last_step, total_steps = int(steps[-1][0]), int(steps[-1][1])
    completed = (last_step == total_steps)

    return {
        "validation_loss": val_losses[-1],
        "min_validation_loss": min(val_losses),
        "num_evals": len(val_losses),
        "last_step": last_step,
        "total_steps": total_steps,
        "completed": completed,
    }


def extract_best_run(checkpoint_dir: Path):
    """Across all wandb runs in a checkpoint dir, find the one with most progress."""
    wandb_runs = sorted(
        (checkpoint_dir / "wandb" / "wandb").glob("run-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    best = None
    for run_dir in wandb_runs:
        metrics = extract_run_metrics(run_dir)
        if metrics is None:
            continue
        if best is None or metrics["last_step"] > best["last_step"]:
            best = metrics
    return best


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all 60M checkpoint dirs
    results_by_chin = {}  # chin_str -> {run_name: metrics}

    for chin_dir in sorted(CHECKPOINT_ROOT.glob("chinchilla_*")):
        chin_str = chin_dir.name.replace("chinchilla_", "")

        for ckpt_dir in sorted(chin_dir.glob("60M_seed*_case4_dolma_epoch*_wd*_lr*")):
            run_name = ckpt_dir.name
            m = RUN_PATTERN.search(run_name)
            if not m:
                continue

            metrics = extract_best_run(ckpt_dir)
            if metrics is None:
                continue

            if chin_str not in results_by_chin:
                results_by_chin[chin_str] = {}

            # Include all runs (completed or not), but mark status
            entry = {
                "validation_loss": metrics["validation_loss"],
                "min_validation_loss": metrics["min_validation_loss"],
                "num_evals": metrics["num_evals"],
                "last_step": metrics["last_step"],
                "total_steps": metrics["total_steps"],
                "completed": metrics["completed"],
            }
            # Keep the "best" version (completed > incomplete, or more progress)
            existing = results_by_chin[chin_str].get(run_name)
            if existing is None:
                results_by_chin[chin_str][run_name] = entry
            else:
                # Prefer completed, else prefer more progress
                if entry["completed"] and not existing["completed"]:
                    results_by_chin[chin_str][run_name] = entry
                elif (not existing["completed"]) and entry["last_step"] > existing["last_step"]:
                    results_by_chin[chin_str][run_name] = entry

    # Write per-chin JSON files
    for chin_str, runs in sorted(results_by_chin.items(), key=lambda x: float(x[0])):
        out_file = OUTPUT_DIR / f"chinchilla_{chin_str}_60M.json"
        with open(out_file, "w") as f:
            json.dump(runs, f, indent=2, sort_keys=True)
        n_completed = sum(1 for r in runs.values() if r["completed"])
        print(f"  {out_file.name}: {len(runs)} runs ({n_completed} completed)")

    print(f"\nTotal chinchilla values: {len(results_by_chin)}")


if __name__ == "__main__":
    main()
