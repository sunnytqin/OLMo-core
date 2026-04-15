#!/usr/bin/env python3
"""
Discover all checkpoints in the barak_lab checkpoint directory,
find the last step for each, validate completeness, and output a manifest JSON.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

CANONICAL_DIRS = [
    "chinchilla_0.05", "chinchilla_0.1", "chinchilla_0.25",
    "chinchilla_0.5", "chinchilla_1", "chinchilla_2", "chinchilla_4",
]

RUN_PATTERN = re.compile(
    r'^(30M|190M|370M)_seed(\d+)_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)$'
)


def find_last_step(run_dir: Path) -> int | None:
    """Find the highest-numbered step directory."""
    steps = []
    for entry in run_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("step"):
            try:
                steps.append(int(entry.name[4:]))
            except ValueError:
                continue
    return max(steps) if steps else None


def is_checkpoint_valid(step_dir: Path) -> bool:
    """Check that model_and_optim/ has actual files."""
    model_dir = step_dir / "model_and_optim"
    if not model_dir.exists():
        return False
    return any(model_dir.iterdir())


def discover(base_dir: str, chinchilla_dirs: list[str] | None = None) -> dict:
    base = Path(base_dir)
    dirs = chinchilla_dirs or CANONICAL_DIRS

    checkpoints = []
    invalid = []

    # First pass: collect all checkpoints and group by (chin, size, epoch)
    # to detect incomplete runs
    group_steps = defaultdict(list)  # (chin, size, epoch) -> [last_step, ...]

    raw_entries = []
    for chin_dir in dirs:
        chin_path = base / chin_dir
        if not chin_path.exists():
            print(f"WARNING: {chin_path} not found, skipping", file=sys.stderr)
            continue

        for run_dir in sorted(chin_path.iterdir()):
            if not run_dir.is_dir():
                continue
            m = RUN_PATTERN.match(run_dir.name)
            if not m:
                continue

            model_size = m.group(1)
            epoch = int(m.group(3))
            wd = m.group(4)
            lr = m.group(5)

            last_step = find_last_step(run_dir)
            if last_step is None:
                invalid.append({"run_name": run_dir.name, "chinchilla_dir": chin_dir,
                                "reason": "no step directories"})
                continue

            step_dir = run_dir / f"step{last_step}"
            if not is_checkpoint_valid(step_dir):
                invalid.append({"run_name": run_dir.name, "chinchilla_dir": chin_dir,
                                "reason": "empty model_and_optim"})
                continue

            group_key = (chin_dir, model_size, epoch)
            group_steps[group_key].append(last_step)

            raw_entries.append({
                "run_name": run_dir.name,
                "chinchilla_dir": chin_dir,
                "model_size": model_size,
                "epoch": epoch,
                "wd": wd,
                "lr": lr,
                "checkpoint_path": str(step_dir),
                "last_step": last_step,
            })

    # Second pass: detect incomplete runs
    # For each group, the expected step count is the mode (most common value)
    group_expected = {}
    for key, steps in group_steps.items():
        # Use the most common step count as expected
        from collections import Counter
        counts = Counter(steps)
        expected = counts.most_common(1)[0][0]
        group_expected[key] = expected

    for entry in raw_entries:
        key = (entry["chinchilla_dir"], entry["model_size"], entry["epoch"])
        expected = group_expected[key]
        # Allow 5% tolerance
        complete = entry["last_step"] >= expected * 0.95
        entry["complete"] = complete
        if not complete:
            entry["expected_step"] = expected
        checkpoints.append(entry)

    # Summary
    n_complete = sum(1 for c in checkpoints if c["complete"])
    n_incomplete = sum(1 for c in checkpoints if not c["complete"])
    by_size = defaultdict(int)
    for c in checkpoints:
        by_size[c["model_size"]] += 1

    manifest = {
        "checkpoints": checkpoints,
        "invalid": invalid,
        "summary": {
            "total_valid": len(checkpoints),
            "complete": n_complete,
            "incomplete": n_incomplete,
            "invalid": len(invalid),
            "by_size": dict(by_size),
        },
    }

    # Print summary
    print(f"Discovered {len(checkpoints)} valid checkpoints "
          f"({n_complete} complete, {n_incomplete} incomplete, "
          f"{len(invalid)} invalid/skipped)", file=sys.stderr)
    for size, count in sorted(by_size.items()):
        print(f"  {size}: {count}", file=sys.stderr)
    if invalid:
        print(f"\nInvalid checkpoints:", file=sys.stderr)
        for inv in invalid:
            print(f"  {inv['chinchilla_dir']}/{inv['run_name']}: {inv['reason']}",
                  file=sys.stderr)
    if n_incomplete:
        print(f"\nIncomplete runs:", file=sys.stderr)
        for c in checkpoints:
            if not c["complete"]:
                print(f"  {c['chinchilla_dir']}/{c['run_name']}: "
                      f"step {c['last_step']} vs expected {c['expected_step']}",
                      file=sys.stderr)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Discover OLMo checkpoints")
    parser.add_argument("--base-dir", type=str,
                        default="/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints",
                        help="Base checkpoint directory")
    parser.add_argument("--output", type=str, default="../results/hparam/manifest.json",
                        help="Output manifest JSON path")
    args = parser.parse_args()

    manifest = discover(args.base_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
