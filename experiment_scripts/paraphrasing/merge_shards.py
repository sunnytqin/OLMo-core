#!/usr/bin/env python3
"""
Merge paraphrased shard .npy files into a single .npy file.

Usage:
    python merge_shards.py --input-dir <shard_dir> --output <output.npy>
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Merge shard .npy files")
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing shard_XXXX.npy files",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for merged .npy file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    shard_files = sorted(input_dir.glob("shard_*.npy"))

    if not shard_files:
        print(f"No shard .npy files found in {input_dir}")
        return

    print(f"Found {len(shard_files)} shard files:")

    # Calculate total size
    total_tokens = 0
    shard_sizes = []
    for f in shard_files:
        arr = np.memmap(f, dtype=np.uint32, mode="r")
        shard_sizes.append(len(arr))
        total_tokens += len(arr)
        print(f"  {f.name}: {len(arr):,} tokens")
        del arr

    print(f"\nTotal tokens: {total_tokens:,}")

    # Create output memmap and copy shard data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing merged file to: {output_path}")
    merged = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=(total_tokens,))

    offset = 0
    for f, size in tqdm(zip(shard_files, shard_sizes), total=len(shard_files), desc="Merging"):
        shard = np.memmap(f, dtype=np.uint32, mode="r")
        merged[offset : offset + size] = shard[:]
        offset += size
        del shard

    merged.flush()
    del merged

    print(f"\nDone! Merged {len(shard_files)} shards → {output_path}")
    print(f"Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    main()
