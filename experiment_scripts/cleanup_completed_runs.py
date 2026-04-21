#!/usr/bin/env python3
"""
Clean up checkpoint directories for confirmed-complete runs.

For each run confirmed complete (via the same logic as build_scaling_manifest.py),
delete all step* directories EXCEPT the final step. Runs that are not confirmed
complete are left fully intact.

Defaults to dry-run. Pass --apply to actually delete.
"""
import argparse
import json
import os
import re
import glob
import shutil
import sys
from pathlib import Path

CHECKPOINT_BASE = "/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints"
BATCH_SIZE = 2097152  # tokens per step

CHINCHILLA_DIRS = [
    "chinchilla_0.05", "chinchilla_0.1", "chinchilla_0.25", "chinchilla_0.5",
    "chinchilla_1", "chinchilla_2", "chinchilla_4", "chinchilla_8", "chinchilla_16",
]
MODEL_SIZES = ["30M", "60M", "190M", "370M"]

RUN_PATTERN = re.compile(
    r"(?P<size>\d+M)_seed42_case4_dolma_epoch(?P<epoch>\d+)_wd(?P<wd>[\d.]+)_lr(?P<lr>[\de.-]+)"
)


def list_step_dirs(run_path: str):
    """Return list of (step_int, dirname) for every stepN directory (excluding step0)."""
    out = []
    for d in os.listdir(run_path):
        if d.startswith("step") and d != "step0":
            try:
                out.append((int(d[4:]), d))
            except ValueError:
                pass
    return out


def get_max_step_dir(run_path: str):
    """Return (max_step, has_model_weights) for a run directory."""
    steps = [s for s, _ in list_step_dirs(run_path)]
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


def is_complete(run_path: str):
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

    return True, max_step, f"step {max_step} (no target, has model)"


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete. Without this flag, runs in dry-run mode.")
    ap.add_argument("--no-size", action="store_true",
                    help="Skip computing freed size (faster on slow filesystems).")
    ap.add_argument("--strip-step0-weights", action="store_true",
                    help="Also remove step0/model_and_optim for completed runs "
                         "(initial random weights, useless). Keeps step0/config.json.")
    args = ap.parse_args()

    mode = "APPLY (DELETING)" if args.apply else "DRY-RUN (no deletes)"
    print(f"Mode: {mode}")
    print(f"Checkpoint base: {CHECKPOINT_BASE}")
    print()

    total_dirs_to_delete = 0
    total_bytes = 0
    runs_cleaned = 0
    runs_skipped_incomplete = 0
    runs_already_clean = 0

    for chin_dir in CHINCHILLA_DIRS:
        base = os.path.join(CHECKPOINT_BASE, chin_dir)
        if not os.path.isdir(base):
            continue

        for model_size in MODEL_SIZES:
            runs = sorted(glob.glob(os.path.join(base, f"{model_size}_*")))
            for run_path in runs:
                run_name = os.path.basename(run_path)
                m = RUN_PATTERN.match(run_name)
                if not m:
                    continue

                complete, max_step, reason = is_complete(run_path)
                if not complete:
                    runs_skipped_incomplete += 1
                    print(f"SKIP (incomplete): {chin_dir}/{run_name} — {reason}")
                    continue

                step_dirs = list_step_dirs(run_path)
                to_delete_dirs = [os.path.join(run_path, d) for s, d in step_dirs if s != max_step]

                step0_weights = os.path.join(run_path, "step0", "model_and_optim")
                if args.strip_step0_weights and os.path.isdir(step0_weights):
                    to_delete_dirs.append(step0_weights)

                if not to_delete_dirs:
                    runs_already_clean += 1
                    continue

                run_freed = 0
                if not args.no_size:
                    for target in to_delete_dirs:
                        run_freed += dir_size_bytes(target)

                total_dirs_to_delete += len(to_delete_dirs)
                total_bytes += run_freed
                runs_cleaned += 1

                size_str = "" if args.no_size else f" ({fmt_bytes(run_freed)})"
                print(f"CLEAN: {chin_dir}/{run_name} — keeping step{max_step}, "
                      f"removing {len(to_delete_dirs)} dirs{size_str}")

                if args.apply:
                    for target in to_delete_dirs:
                        try:
                            shutil.rmtree(target)
                        except OSError as e:
                            print(f"  ERROR removing {target}: {e}", file=sys.stderr)

    print()
    print("=" * 60)
    print(f"Runs to clean:        {runs_cleaned}")
    print(f"Runs already clean:   {runs_already_clean}")
    print(f"Runs skipped (incomplete): {runs_skipped_incomplete}")
    print(f"Step dirs to remove:  {total_dirs_to_delete}")
    if not args.no_size:
        print(f"Space to free:        {fmt_bytes(total_bytes)}")
    print("=" * 60)
    if not args.apply:
        print("\nDry-run only. Re-run with --apply to actually delete.")


if __name__ == "__main__":
    main()
