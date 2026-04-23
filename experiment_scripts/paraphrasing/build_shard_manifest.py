#!/usr/bin/env python3
"""
Build a shard manifest: for each train_*.npy in the resharded dir, record
num_docs (via EOS count) and num_tokens (via file size). This is the minimal
info the paraphrase prefix-slicing logic needs.

The training shards are nested prefixes of the same globally-shuffled
doc order, so knowing num_docs per shard lets us slice a paraphrased
superset shard down to any smaller shard at training time.

Output: <resharded_dir>/shard_manifest.json
    {
      "train_0.03B.npy":   {"num_docs": 19658,    "num_tokens": 30000170},
      "train_0.6B.npy":    {"num_docs": 394812,   "num_tokens": 600003444},
      ...
    }
"""

import argparse
import json
from pathlib import Path

import numpy as np

EOS_TOKEN_ID = 100257


def count_eos_chunked(path: Path, chunk: int = 500_000_000) -> tuple[int, int]:
    """Chunked EOS count + total token count for one .npy."""
    arr = np.memmap(path, dtype=np.uint32, mode="r")
    total = len(arr)
    eos = 0
    for i in range(0, total, chunk):
        seg = np.asarray(arr[i : i + chunk])
        eos += int((seg == EOS_TOKEN_ID).sum())
    del arr
    return eos, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Dir containing train_*.npy")
    p.add_argument("--output", default=None, help="Output JSON path (default: <data-dir>/shard_manifest.json)")
    p.add_argument("--chunk", type=int, default=500_000_000, help="Chunk size in tokens (default 500M)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output) if args.output else data_dir / "shard_manifest.json"

    npy_files = sorted(data_dir.glob("train_*.npy"))
    print(f"Scanning {len(npy_files)} .npy files in {data_dir}")

    manifest: dict = {}
    for i, f in enumerate(npy_files, 1):
        n_docs, n_tok = count_eos_chunked(f, args.chunk)
        manifest[f.name] = {"num_docs": n_docs, "num_tokens": n_tok}
        print(f"  [{i:>2}/{len(npy_files)}] {f.name:<28}  docs={n_docs:>12,}  tokens={n_tok/1e9:>7.4f}B")

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
