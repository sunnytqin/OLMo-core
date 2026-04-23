#!/usr/bin/env python3
"""
Build a shard_manifest.json for the dolma resharded corpus.

For each train_*.npy shard, records:
  - num_tokens : total token count (file size / 4)
  - num_docs   : count of EOS tokens (100257 for dolma2-tokenizer)

This manifest is the minimum info needed for the paraphrase pipeline's
"nested-prefix" slicing: given a paraphrase of the largest shard, we slice
the first `num_docs(smaller_shard)` paraphrased documents to get that
smaller shard's paraphrase.

Runs chunked EOS counting so it's safe on login nodes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

EOS_TOKEN_ID = np.uint32(100257)


def count_eos(path: Path, chunk: int = 200_000_000) -> tuple[int, int]:
    """Return (num_tokens, num_docs) for a flat uint32 .npy file, chunked to bound memory."""
    arr = np.memmap(path, dtype=np.uint32, mode="r")
    n = len(arr)
    docs = 0
    for i in range(0, n, chunk):
        seg = np.asarray(arr[i : i + chunk])
        docs += int((seg == EOS_TOKEN_ID).sum())
    return n, docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        default="/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer",
        help="Directory containing train_*.npy files",
    )
    p.add_argument("--output", default=None, help="Output JSON path (default: <dir>/shard_manifest.json)")
    args = p.parse_args()

    data_dir = Path(args.dir)
    output = Path(args.output) if args.output else data_dir / "shard_manifest.json"

    npy_files = sorted(data_dir.glob("train_*.npy"))
    print(f"Scanning {len(npy_files)} shards in {data_dir}")

    manifest = {}
    t0 = time.time()
    for i, path in enumerate(npy_files, 1):
        ts = time.time()
        n_tok, n_docs = count_eos(path)
        elapsed = time.time() - ts
        manifest[path.name] = {"num_tokens": n_tok, "num_docs": n_docs}
        print(f"  [{i:>2}/{len(npy_files)}] {path.name:<28} {n_tok/1e9:>7.3f}B tok  {n_docs:>10,} docs  ({elapsed:.1f}s)")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    with open(output, "w") as f:
        json.dump({"shards": manifest, "eos_token_id": int(EOS_TOKEN_ID), "dir": str(data_dir)}, f, indent=2)
    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()
