#!/usr/bin/env python3
"""
Integrity check for every paraphrased seed directory.

Verifies, per shard:
  - .npy + .jsonl both exist
  - .npy file size is in the expected range (200-350 MB; typical ~283 MB)
  - .npy is not all zeros (read first 1000 tokens, check non-zero ratio)
  - .npy EOS count is in the expected range (typical ~152,085; min sanity: >100k)
  - .jsonl line count matches expected source-doc count (152,085 or 152,097 for shard 31)
  - .npy total tokens is reasonable

Skips strict checks for tasks that are currently running (passed via --skip arg).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

EOS_TOKEN_ID = 100257
DTYPE = np.uint32
ROOT = Path(
    "/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/"
    "resharded/allenai/dolma2-tokenizer/paraphrased"
)

# Expected ranges based on inspection of known-good seed 1
EXPECTED_SHARDS = 32
DOCS_PER_FULL_SHARD = 152_085   # shards 0..30
DOCS_LAST_SHARD = 152_097       # shard 31

# Reference (from seed 1): 280-286 MB per shard
SIZE_MIN_MB = 200
SIZE_MAX_MB = 400


def check_shard(seed_dir: Path, shard_idx: int) -> dict:
    """Return a dict {ok: bool, issues: [str], stats: {...}}."""
    name = f"shard_{shard_idx:04d}"
    npy = seed_dir / f"{name}.npy"
    jsonl = seed_dir / f"{name}.jsonl"
    issues = []
    stats = {}

    if not jsonl.exists():
        issues.append("missing_jsonl")
    if not npy.exists():
        issues.append("missing_npy")
    if issues:
        return {"ok": False, "issues": issues, "stats": stats}

    # File-size sanity
    sz = npy.stat().st_size
    stats["size_mb"] = sz / 1e6
    if sz < SIZE_MIN_MB * 1e6:
        issues.append(f"size_too_small({sz/1e6:.1f}MB)")
    if sz > SIZE_MAX_MB * 1e6:
        issues.append(f"size_too_large({sz/1e6:.1f}MB)")

    # mmap and check contents
    try:
        arr = np.memmap(npy, dtype=DTYPE, mode="r")
    except Exception as e:
        issues.append(f"memmap_failed({e})")
        return {"ok": False, "issues": issues, "stats": stats}

    n_tok = arr.shape[0]
    stats["tokens"] = int(n_tok)

    # Spot-check first 1000 tokens — if they're all zero something is very wrong
    head = arr[:1000]
    nonzero_ratio = float(np.count_nonzero(head) / max(1, head.size))
    stats["head_nonzero_ratio"] = nonzero_ratio
    if nonzero_ratio < 0.5:
        issues.append(f"head_mostly_zero({nonzero_ratio:.2f})")

    # EOS count = number of paraphrased docs in the shard
    n_eos = int(np.count_nonzero(arr == EOS_TOKEN_ID))
    stats["eos_count"] = n_eos
    if n_eos < 100_000:
        issues.append(f"eos_count_too_low({n_eos})")

    # JSONL line count
    try:
        with open(jsonl) as f:
            n_lines = sum(1 for _ in f)
    except Exception as e:
        issues.append(f"jsonl_read_failed({e})")
        n_lines = 0
    stats["jsonl_lines"] = n_lines
    expected = DOCS_LAST_SHARD if shard_idx == 31 else DOCS_PER_FULL_SHARD
    if n_lines != expected:
        issues.append(f"jsonl_lines_mismatch(got={n_lines}, expected={expected})")

    return {"ok": not issues, "issues": issues, "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="",
                    help="comma-separated seed:shard pairs to skip strict check (still report stats). "
                         "e.g. '28:0,28:1,30:6'")
    ap.add_argument("--seeds", default="1-32", help="seed range, e.g. '1-32' or '17-32' or '1,5,10'")
    args = ap.parse_args()

    # Parse seeds
    if "," in args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif "-" in args.seeds:
        a, b = map(int, args.seeds.split("-"))
        seeds = list(range(a, b + 1))
    else:
        seeds = [int(args.seeds)]

    # Parse skip
    skip = set()
    if args.skip:
        for tok in args.skip.split(","):
            tok = tok.strip()
            if tok:
                s, sh = tok.split(":")
                skip.add((int(s), int(sh)))

    overall_ok = True
    for seed in seeds:
        seed_dir = ROOT / f"train_7.4B_smollm2_mixed_seed{seed}"
        if not seed_dir.exists():
            print(f"seed {seed}: DIRECTORY MISSING")
            overall_ok = False
            continue

        bad_shards = []
        all_sizes = []
        all_tokens = []
        all_eos = []
        skipped_shards = []
        for sh in range(EXPECTED_SHARDS):
            if (seed, sh) in skip:
                # Still check, but don't count as bad
                res = check_shard(seed_dir, sh)
                if res["stats"]:
                    all_sizes.append(res["stats"].get("size_mb", 0))
                    all_tokens.append(res["stats"].get("tokens", 0))
                    all_eos.append(res["stats"].get("eos_count", 0))
                skipped_shards.append(sh)
                continue
            res = check_shard(seed_dir, sh)
            if res["stats"]:
                all_sizes.append(res["stats"].get("size_mb", 0))
                all_tokens.append(res["stats"].get("tokens", 0))
                all_eos.append(res["stats"].get("eos_count", 0))
            if not res["ok"]:
                bad_shards.append((sh, res["issues"], res["stats"]))

        # Per-seed summary
        n_ok = EXPECTED_SHARDS - len(bad_shards) - len(skipped_shards)
        n_total_tok = sum(all_tokens)
        if bad_shards:
            overall_ok = False
            tag = "FAIL"
        else:
            tag = "OK"
        skip_tag = f", skipped {len(skipped_shards)}" if skipped_shards else ""
        print(
            f"seed {seed:>2}: [{tag}]  {n_ok}/{EXPECTED_SHARDS} clean shards{skip_tag}, "
            f"total={n_total_tok/1e9:.2f}B tok, "
            f"size_range={min(all_sizes):.0f}-{max(all_sizes):.0f}MB" if all_sizes else
            f"seed {seed:>2}: [{tag}]  no shards"
        )
        if bad_shards:
            for sh, issues, stats in bad_shards:
                print(f"   shard_{sh:04d}: {issues}  stats={stats}")
        if skipped_shards:
            print(f"   (skipped — currently running: {skipped_shards})")

    print()
    print("=" * 60)
    print("OVERALL OK" if overall_ok else "OVERALL FAIL — see above")


if __name__ == "__main__":
    main()
