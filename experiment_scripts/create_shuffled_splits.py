#!/usr/bin/env python3
"""
Create Sequential Train/Validation Splits from DCLM Data (streaming, multi-file)

GOAL
----
Create sequential data splits with nested training sets:
  - validation.npy: 500M tokens (distinct from training)
  - train_0.3B.npy: 0.3B tokens (first portion of training)
  - train_0.6B.npy: 0.6B tokens (contains train_0.3B)
  - train_2.4B.npy: 2.4B tokens (contains train_0.6B)
  - train_4.8B.npy: 4.8B tokens (contains train_2.4B)
  - train_9.6B.npy: 9.6B tokens (contains train_4.8B)

KEY RULES
---------
- NO SHUFFLING (preserve coherence)
- Purely sequential slices from the concatenation of input files
- Outputs are RAW binary uint32 (no .npy header), like before

INPUTS
------
Either:
  --input-files /path/part-000-00000.npy /path/part-000-00001.npy ...
or:
  --input-dir /path/to/dir [--glob "part-*.npy"]

Notes:
- Files are treated as sequences of uint32 tokens concatenated in lexicographic order.
- Supports both raw binary and .npy-with-header inputs:
  - If file starts with NPY magic, we load with np.load(..., mmap_mode='r')
  - Else we fall back to np.memmap(..., dtype=np.uint32)

OUTPUTS
-------
Output dir (required): --output-dir /path/to/resharded
  - validation.npy      (500M tokens)
  - train_0.3B.npy      (300M tokens)
  - train_0.6B.npy      (600M tokens)
  - train_2.4B.npy      (2.4B tokens)
  - train_4.8B.npy      (4.8B tokens)
  - train_9.6B.npy      (9.6B tokens)

PERF / MEMORY
-------------
- Streaming copy in chunks (default 100M tokens ~= 400MB per step)
- Minimal RAM use; scales beyond 32GB nodes
"""

import argparse
import os
from pathlib import Path
import numpy as np

NPY_MAGIC_PREFIX = b"\x93NUMPY"

def is_npy_with_header(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(6)
    return head.startswith(NPY_MAGIC_PREFIX)

def open_token_array(path: Path):
    """
    Return an array-like (memmap) of dtype=uint32 for the file.
    Supports:
      - Raw binary uint32 (no header) via np.memmap
      - .npy files with header via np.load(..., mmap_mode='r')
    """
    if is_npy_with_header(path):
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != np.uint32:
            raise ValueError(f"{path} has dtype {arr.dtype}, expected uint32")
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr
    else:
        # raw binary
        return np.memmap(path, dtype=np.uint32, mode="r")

def gather_sources(files):
    sources = []
    sizes = []
    for p in files:
        arr = open_token_array(p)
        sources.append(arr)
        sizes.append(arr.size)
    cum = np.cumsum([0] + sizes)  # prefix sums, len = n_files+1
    total = cum[-1]
    return sources, sizes, cum, total

def copy_range_to_memmap(sources, sizes, cum, start, length, out_path: Path, chunk_tokens: int):
    """
    Copy [start, start+length) tokens from the virtual concatenation of `sources`
    into a raw-binary uint32 file at out_path using a writable memmap.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(length,))
    remaining = length
    gpos = start
    dpos = 0

    # Precompute which source we start in
    # Find largest i such that cum[i] <= gpos
    src_idx = int(np.searchsorted(cum, gpos, side="right") - 1)

    while remaining > 0:
        # Current source bounds in global coordinates: [cum[src_idx], cum[src_idx]+sizes[src_idx])
        src_start_global = cum[src_idx]
        src_size = sizes[src_idx]
        in_src_pos = gpos - src_start_global
        can_take = min(remaining, src_size - in_src_pos, chunk_tokens)

        if can_take <= 0:
            # Move to next source
            src_idx += 1
            if src_idx >= len(sources):
                break
            continue

        # Slice from current source
        src_arr = sources[src_idx]
        dst[dpos:dpos+can_take] = src_arr[in_src_pos:in_src_pos+can_take]

        # Advance pointers
        remaining -= can_take
        gpos += can_take
        dpos += can_take

        # If finished this source, advance
        if in_src_pos + can_take >= src_size and remaining > 0:
            src_idx += 1
            if src_idx >= len(sources) and remaining > 0:
                raise RuntimeError("Ran out of source tokens unexpectedly.")

    dst.flush()
    del dst  # close memmap

def natural_sort_key(p: Path):
    # Sort parts nicely: part-000-00000 before part-000-00001, etc.
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(p))]

def main():
    parser = argparse.ArgumentParser(description="Create sequential data splits (streaming, multi-file).")
    parser.add_argument("--input-files", type=str, nargs="*", default=None,
                        help="Explicit list of input files in order (raw uint32 or .npy).")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing input parts (used if --input-files not given).")
    parser.add_argument("--glob", type=str, default="part-*.npy",
                        help="Glob to match inside --input-dir (default: part-*.npy).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for resharded files.")
    parser.add_argument("--chunk-tokens", type=int, default=100_000_000,
                        help="Streaming chunk size in tokens (default: 100M ~= 400MB).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define validation and training shard configurations
    # Each training shard is nested (larger shards contain all smaller ones)
    val_size = 500_000_000  # 0.5B

    train_shards = [
        {"name": "train_0.3B.npy", "tokens": 300_000_000},    # 0.3B
        {"name": "train_0.6B.npy", "tokens": 600_000_000},    # 0.6B
        {"name": "train_1.2B.npy", "tokens": 1_200_000_000},  # 1.2B
        {"name": "train_2.4B.npy", "tokens": 2_400_000_000},  # 2.4B
        {"name": "train_4.8B.npy", "tokens": 4_800_000_000},  # 4.8B
        {"name": "train_9.6B.npy", "tokens": 9_600_000_000},  # 9.6B
    ]

    max_train_tokens = train_shards[-1]["tokens"]
    total_needed = val_size + max_train_tokens
    print(f"Total tokens needed: {total_needed:,} ({total_needed/1e9:.2f}B)")
    print(f"Disk for outputs ~ {total_needed*4/1e9:.2f} GB (validation + largest training shard)")
    print(f"Chunk size: {args.chunk_tokens:,} tokens (~{args.chunk_tokens*4/1e6:.1f} MB)\n")

    # Resolve inputs
    files = []
    if args.input_files:
        files = [Path(p) for p in args.input_files]
    else:
        if not args.input_dir:
            raise ValueError("Provide either --input-files ... or --input-dir DIR")
        d = Path(args.input_dir)
        files = sorted(d.glob(args.glob), key=natural_sort_key)
        if not files:
            raise ValueError(f"No files matched {args.glob} under {d}")
    print("Input files (in order):")
    for p in files:
        print("  -", p)

    # Open sources and compute total tokens
    print("\nIndexing sources...")
    sources, sizes, cum, total_tokens = gather_sources(files)
    print(f"Total available tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    if total_tokens < total_needed:
        raise ValueError(f"Not enough tokens: need {total_needed:,}, have {total_tokens:,}")

    # Output paths
    val_path = out_dir / "validation.npy"
    train_paths = [out_dir / shard["name"] for shard in train_shards]
    all_output_paths = [val_path] + train_paths

    # Prepare outputs (fail fast if exist)
    for p in all_output_paths:
        if p.exists() and not args.overwrite:
            print(f"Will not overwrite existing file: {p} (use --overwrite to replace)")

    # Remove (if overwrite)
    if args.overwrite:
        for p in all_output_paths:
            if p.exists():
                p.unlink()

    # Layout (global positions)
    train_base_offset = val_size

    # Write validation split
    total_shards = len(train_shards) + 1
    print(f"\n[1/{total_shards}] Writing validation ({val_size/1e9:.1f}B tokens)...")
    copy_range_to_memmap(sources, sizes, cum,
                         start=0, length=val_size,
                         out_path=val_path, chunk_tokens=args.chunk_tokens)
    print(f"      ✓ validation saved to {val_path}")

    # Write training shards (nested structure - each contains all smaller ones)
    for idx, shard in enumerate(train_shards, start=2):
        shard_name = shard["name"]
        shard_tokens = shard["tokens"]
        shard_path = out_dir / shard_name

        # Describe nesting
        if idx == 2:
            nested_info = ""
        else:
            prev_shard = train_shards[idx - 3]["name"]
            nested_info = f" (contains {prev_shard.replace('.npy', '')})"

        print(f"\n[{idx}/{total_shards}] Writing {shard_name.replace('.npy', '')} ({shard_tokens/1e9:.1f}B tokens)...{nested_info}")
        copy_range_to_memmap(sources, sizes, cum,
                             start=train_base_offset, length=shard_tokens,
                             out_path=shard_path, chunk_tokens=args.chunk_tokens)
        print(f"      ✓ {shard_name.replace('.npy', '')} saved to {shard_path}")

    # Final report
    print("\n" + "="*80)
    print("SUCCESS! All splits created:")
    size_gb = (val_path.stat().st_size / 1e9) if val_path.exists() else 0.0
    print(f"  {val_path.name}: {val_size:,} tokens ({size_gb:.2f} GB)")

    for shard, shard_path in zip(train_shards, train_paths):
        size_gb = (shard_path.stat().st_size / 1e9) if shard_path.exists() else 0.0
        print(f"  {shard['name']}: {shard['tokens']:,} tokens ({size_gb:.2f} GB)")

    print("\nNested structure:")
    shard_names = [s["name"].replace(".npy", "") for s in train_shards]
    nested_str = " ⊇ ".join(reversed(shard_names))
    print(f"  {nested_str}; all training sets exclude validation.")
    print("="*80)

if __name__ == "__main__":
    main()