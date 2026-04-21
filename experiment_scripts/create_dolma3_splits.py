#!/usr/bin/env python3
"""
Create Document-Sampled Train/Validation Splits from Dolma2 Data

GOAL
----
Uniformly sample documents from the entire tokenized corpus and create nested
training splits + a validation set.  Smaller training splits are strict subsets
of larger ones.  Everything is reproducible via --seed.

SPLITS CREATED
--------------
  validation.npy     : ~500M tokens  (randomly sampled, disjoint from train)
  train_0.03B.npy    : ~30M   tokens
  train_0.06B.npy    : ~60M   tokens  (contains train_0.03B)
  train_0.15B.npy    : ~150M  tokens  (contains train_0.06B)
  train_0.3B.npy     : ~300M  tokens  (contains train_0.15B)
  train_0.6B.npy     : ~600M  tokens  (contains train_0.3B)
  train_1.2B.npy     : ~1.2B  tokens  (contains train_0.6B)
  train_2.4B.npy     : ~2.4B  tokens  (contains train_1.2B)
  train_4.8B.npy     : ~4.8B  tokens  (contains train_2.4B)
  train_9.6B.npy     : ~9.6B  tokens  (contains train_4.8B)

Actual token counts may slightly exceed targets because splits always use
complete documents (no mid-document cuts).

WHY SCAN ALL FILES
------------------
A naive approach of randomly selecting a subset of files before sampling
documents from them would introduce a bias: documents in under-sampled files
would never appear in the splits.  To get truly uniform sampling we must first
build an index of *all* documents across *all* files (mirroring how
NumpyFSLDataLoader._build_global_indices works over the full dataset), then
draw from that global pool.

ALGORITHM
---------
1. Discover all .npy files under --data-dir (recursive, sorted for determinism).
2. Scan every file to find EOS token (100257 = <|endoftext|>) positions and
   record (file_index, start, end) for each document.  Results are cached.
3. Shuffle all documents globally with the seed.
4. Assign shuffled documents greedily to validation then each training shard.
   Training shards take nested prefixes of the same shuffled order, guaranteeing
   smaller shards ⊆ larger shards.
5. Write each split as a flat raw uint32 binary file (no numpy header).

CACHING
-------
Step 2 reads the entire corpus (~600 GB for 150B tokens) and is the only
expensive part.  Results are saved to <output-dir>/doc_index.npz so that
every subsequent run skips straight to step 3.  Use --rebuild-index to force
a rebuild (e.g. if the source data changes).

OUTPUT FORMAT
-------------
Each output file is a flat raw binary array of uint32 tokens, readable with:
    np.memmap(path, dtype=np.uint32, mode='r')

EXAMPLE
-------
python experiment_scripts/create_doc_sampled_splits.py \\
    --data-dir /n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer \\
    --output-dir /path/to/output \\
    --seed 42
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# <|endoftext|> for allenai/dolma2-tokenizer
DEFAULT_EOS_TOKEN = 100257

VAL_TARGET = 500_000_000  # 0.5 B tokens

# ---------------------------------------------------------------------------
# Per-model shard definitions
#
# TTP (tokens-to-parameters ratio): TTP=20 is chinchilla-optimal (chin=1).
# Each model lists the TTP ladder it needs, from TTP=1 (under-trained) up to
# TTP=320 (16x chinchilla).  Token counts must be exact integers so the
# training-script lookup table can match them precisely.
#
# Splits shared across models (e.g. 0.6B is 30M TTP=20 AND 60M TTP=10)
# appear only once in the output — the merge step deduplicates by token count.
# ---------------------------------------------------------------------------
_PER_MODEL_SHARDS = {
    "14M": {   # non-embedding params ≈ 14M; chin unit = 280M tokens
        "train_0.014B.npy": (   14_000_000,   1),   # TTP=1   chin=0.05
        "train_0.028B.npy": (   28_000_000,   2),   # TTP=2   chin=0.1
        "train_0.07B.npy":  (   70_000_000,   5),   # TTP=5   chin=0.25
        "train_0.14B.npy":  (  140_000_000,  10),   # TTP=10  chin=0.5
        "train_0.28B.npy":  (  280_000_000,  20),   # TTP=20  chin=1
        "train_0.56B.npy":  (  560_000_000,  40),   # TTP=40  chin=2
        "train_1.12B.npy":  (1_120_000_000,  80),   # TTP=80  chin=4
        "train_2.24B.npy":  (2_240_000_000, 160),   # TTP=160 chin=8
        "train_4.48B.npy":  (4_480_000_000, 320),   # TTP=320 chin=16
        "train_8.96B.npy":  (8_960_000_000, 640),   # TTP=640 chin=32
    },
    "30M": {   # non-embedding params ≈ 30M; chin unit = 600M tokens
        "train_0.03B.npy":  (   30_000_000,   1),   # TTP=1
        "train_0.06B.npy":  (   60_000_000,   2),   # TTP=2
        "train_0.15B.npy":  (  150_000_000,   5),   # TTP=5
        "train_0.3B.npy":   (  300_000_000,  10),   # TTP=10
        "train_0.6B.npy":   (  600_000_000,  20),   # TTP=20  chin=1
        "train_1.2B.npy":   (1_200_000_000,  40),   # TTP=40  chin=2
        "train_2.4B.npy":   (2_400_000_000,  80),   # TTP=80  chin=4
        "train_4.8B.npy":   (4_800_000_000, 160),   # TTP=160 chin=8
        "train_9.6B.npy":   (9_600_000_000, 320),   # TTP=320 chin=16
    },
    "60M": {   # non-embedding params ≈ 60M; chin unit = 1200M tokens
        "train_0.06B.npy":  (    60_000_000,   1),  # TTP=1
        "train_0.12B.npy":  (   120_000_000,   2),  # TTP=2
        "train_0.3B.npy":   (   300_000_000,   5),  # TTP=5
        "train_0.6B.npy":   (   600_000_000,  10),  # TTP=10
        "train_1.2B.npy":   ( 1_200_000_000,  20),  # TTP=20  chin=1
        "train_2.4B.npy":   ( 2_400_000_000,  40),  # TTP=40  chin=2
        "train_4.8B.npy":   ( 4_800_000_000,  80),  # TTP=80  chin=4
        "train_9.6B.npy":   ( 9_600_000_000, 160),  # TTP=160 chin=8
        "train_19.2B.npy":  (19_200_000_000, 320),  # TTP=320 chin=16
    },
    "100M": {  # non-embedding params ≈ 100M; chin unit = 2000M tokens
        "train_0.1B.npy":   (   100_000_000,   1),  # TTP=1   chin=0.05
        "train_0.2B.npy":   (   200_000_000,   2),  # TTP=2   chin=0.1
        "train_0.5B.npy":   (   500_000_000,   5),  # TTP=5   chin=0.25
        "train_1.0B.npy":   ( 1_000_000_000,  10),  # TTP=10  chin=0.5
        "train_2.0B.npy":   ( 2_000_000_000,  20),  # TTP=20  chin=1
        "train_4.0B.npy":   ( 4_000_000_000,  40),  # TTP=40  chin=2
        "train_8.0B.npy":   ( 8_000_000_000,  80),  # TTP=80  chin=4
        "train_16.0B.npy":  (16_000_000_000, 160),  # TTP=160 chin=8
        "train_32.0B.npy":  (32_000_000_000, 320),  # TTP=320 chin=16
    },
    "190M": {  # non-embedding params ≈ 190M; chin unit = 3800M tokens
        "train_0.19B.npy":  (   190_000_000,   1),  # TTP=1
        "train_0.38B.npy":  (   380_000_000,   2),  # TTP=2
        "train_0.95B.npy":  (   950_000_000,   5),  # TTP=5
        "train_1.9B.npy":   ( 1_900_000_000,  10),  # TTP=10
        "train_3.8B.npy":   ( 3_800_000_000,  20),  # TTP=20  chin=1
        "train_7.6B.npy":   ( 7_600_000_000,  40),  # TTP=40  chin=2
        "train_15.2B.npy":  (15_200_000_000,  80),  # TTP=80  chin=4
        "train_30.4B.npy":  (30_400_000_000, 160),  # TTP=160 chin=8
        "train_60.8B.npy":  (60_800_000_000, 320),  # TTP=320 chin=16
    },
    "370M": {  # non-embedding params ≈ 370M; chin unit = 7400M tokens
        "train_0.37B.npy":  (   370_000_000,   1),  # TTP=1
        "train_0.74B.npy":  (   740_000_000,   2),  # TTP=2
        "train_1.85B.npy":  ( 1_850_000_000,   5),  # TTP=5
        "train_3.7B.npy":   ( 3_700_000_000,  10),  # TTP=10
        "train_7.4B.npy":   ( 7_400_000_000,  20),  # TTP=20  chin=1
        "train_14.8B.npy":  (14_800_000_000,  40),  # TTP=40  chin=2
        "train_29.6B.npy":  (29_600_000_000,  80),  # TTP=80  chin=4
        "train_59.2B.npy":  (59_200_000_000, 160),  # TTP=160 chin=8
    },
    "600M": {  # non-embedding params ≈ 600M; chin unit = 12000M tokens
        "train_0.6B.npy":   (   600_000_000,   1),  # TTP=1   chin=0.05
        "train_1.2B.npy":   ( 1_200_000_000,   2),  # TTP=2   chin=0.1
        "train_3.0B.npy":   ( 3_000_000_000,   5),  # TTP=5   chin=0.25
        "train_6.0B.npy":   ( 6_000_000_000,  10),  # TTP=10  chin=0.5
        "train_12.0B.npy":  (12_000_000_000,  20),  # TTP=20  chin=1
        "train_24.0B.npy":  (24_000_000_000,  40),  # TTP=40  chin=2
        "train_48.0B.npy":  (48_000_000_000,  80),  # TTP=80  chin=4
    },
}

# ---------------------------------------------------------------------------
# Chunking: shards at or above CHUNK_THRESHOLD are written as multiple part
# files (train_XB.part00.npy, train_XB.part01.npy, ...) so each part fits in
# a reasonable SLURM wallclock.  Parts are contiguous slices of the same
# globally-shuffled doc_order, so nested-prefix invariants are preserved
# across shards of different sizes.
# ---------------------------------------------------------------------------
CHUNK_THRESHOLD = 20_000_000_000   # shards >= 20B get chunked into parts
PART_TARGET     =  5_000_000_000   # target tokens per part (→ 24B: 5 parts, 29.6B: 6, 59.2B: 12, etc.)


def num_parts_for(target_tokens: int) -> int:
    """How many part files a shard of `target_tokens` should be written as."""
    if target_tokens < CHUNK_THRESHOLD:
        return 1
    return max(1, int(np.ceil(target_tokens / PART_TARGET)))


def part_filenames(shard_name: str, n_parts: int) -> list:
    """Given 'train_29.6B.npy' and n_parts=6, return 6 'train_29.6B.partNN.npy' names."""
    if n_parts == 1:
        return [shard_name]
    stem = shard_name[: -len(".npy")]
    return [f"{stem}.part{i:02d}.npy" for i in range(n_parts)]

# Merge: deduplicate by token count, sort ascending.
# Smaller shards ⊆ larger shards because write_split always takes the first N
# documents of the same globally-shuffled pool.
_merged: dict = {}
for _entries in _PER_MODEL_SHARDS.values():
    for _name, (_tokens, _) in _entries.items():
        _merged[_tokens] = _name  # last writer wins (names are identical for shared sizes)

TRAIN_SHARDS = [
    {"name": name, "tokens": tokens}
    for tokens, name in sorted(_merged.items())
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _natural_sort_key(p: Path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(p))]


def discover_files(root: Path) -> list:
    """Recursively find all .npy files under root, sorted for determinism."""
    files = sorted(root.rglob("*.npy"), key=_natural_sort_key)
    if not files:
        raise ValueError(f"No .npy files found under {root}")
    return files


def _file_list_hash(files: list) -> str:
    """Stable hash of (path, size) pairs — used to detect corpus changes."""
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f}:{f.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Document index building  (the expensive one-time step)
# ---------------------------------------------------------------------------

def build_doc_index(files: list, eos_token: int) -> tuple:
    """
    Scan every file in `files` and record one span per document.

    A document is a contiguous token span ending with EOS (inclusive).
    Tokens after the last EOS in a file are kept as a trailing partial doc
    so no data is silently dropped.

    This mirrors how NumpyFSLDataLoader covers the full dataset: it never
    restricts to a subset of files — it indexes every token.

    Returns
    -------
    doc_file_idx : int32 ndarray  – index into `files` for each document
    doc_starts   : int64 ndarray  – first token offset in file (inclusive)
    doc_ends     : int64 ndarray  – last token offset + 1  (exclusive)
    """
    file_idx_parts = []
    starts_parts   = []
    ends_parts     = []

    t0 = time.time()
    total_tokens_scanned = 0

    for file_idx, path in enumerate(files):
        file_tokens = path.stat().st_size // 4
        if file_tokens == 0:
            continue  # skip empty files

        arr = np.memmap(path, dtype=np.uint32, mode="r")
        n   = len(arr)

        eos_pos = np.where(arr == eos_token)[0].astype(np.int64)
        del arr  # release OS pages as soon as possible

        if len(eos_pos) == 0:
            # Whole file is one EOS-less document
            starts = np.array([0],  dtype=np.int64)
            ends   = np.array([n],  dtype=np.int64)
        else:
            doc_ends_arr   = eos_pos + 1                                      # end of each complete doc
            doc_starts_arr = np.concatenate([[np.int64(0)], doc_ends_arr[:-1]])

            if int(doc_ends_arr[-1]) < n:
                # Trailing tokens after last EOS → include as partial doc
                starts = np.concatenate([doc_starts_arr, [doc_ends_arr[-1]]])
                ends   = np.concatenate([doc_ends_arr,   [np.int64(n)]])
            else:
                starts = doc_starts_arr
                ends   = doc_ends_arr

        # Drop any zero-length spans (defensive)
        valid  = ends > starts
        starts = starts[valid]
        ends   = ends[valid]

        n_docs = len(starts)
        file_idx_parts.append(np.full(n_docs, file_idx, dtype=np.int32))
        starts_parts.append(starts)
        ends_parts.append(ends)

        total_tokens_scanned += n

        if (file_idx + 1) % 100 == 0 or file_idx + 1 == len(files):
            elapsed   = time.time() - t0
            total_docs = sum(len(x) for x in file_idx_parts)
            rate      = total_tokens_scanned / max(elapsed, 1e-6) / 1e9
            print(
                f"  [{file_idx+1:>5}/{len(files)} files | "
                f"{total_tokens_scanned/1e9:>7.2f}B tokens | "
                f"{total_docs:>12,} docs | "
                f"{rate:.2f}B tok/s]",
                flush=True,
            )

    return (
        np.concatenate(file_idx_parts),
        np.concatenate(starts_parts),
        np.concatenate(ends_parts),
    )


# ---------------------------------------------------------------------------
# Index cache helpers
# ---------------------------------------------------------------------------

def _cache_valid(out_dir: Path, file_list_hash: str, eos_token: int) -> bool:
    meta_path = out_dir / "doc_index_meta.json"
    npz_path  = out_dir / "doc_index.npz"
    if not meta_path.exists() or not npz_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    return (
        meta.get("file_list_hash") == file_list_hash
        and meta.get("eos_token")   == eos_token
    )


def save_index(out_dir: Path, doc_file_idx, doc_starts, doc_ends,
               file_list_hash: str, files: list, eos_token: int):
    np.savez_compressed(
        out_dir / "doc_index.npz",
        doc_file_idx=doc_file_idx,
        doc_starts=doc_starts,
        doc_ends=doc_ends,
    )
    meta = {
        "file_list_hash": file_list_hash,
        "eos_token":      eos_token,
        "n_docs":         int(len(doc_file_idx)),
        "n_files":        len(files),
        "files":          [str(f) for f in files],
    }
    with open(out_dir / "doc_index_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_index(out_dir: Path):
    cache = np.load(out_dir / "doc_index.npz")
    with open(out_dir / "doc_index_meta.json") as f:
        meta = json.load(f)
    files = [Path(p) for p in meta["files"]]
    return cache["doc_file_idx"], cache["doc_starts"], cache["doc_ends"], files


# ---------------------------------------------------------------------------
# Split boundary helpers
# ---------------------------------------------------------------------------

def find_doc_cutoff(
    doc_order:    np.ndarray,
    doc_starts:   np.ndarray,
    doc_ends:     np.ndarray,
    target_tokens: int,
) -> tuple:
    """
    Return (n_docs, actual_tokens): the smallest prefix of doc_order whose
    cumulative token count first reaches >= target_tokens.
    """
    lengths = (doc_ends[doc_order] - doc_starts[doc_order]).astype(np.int64)
    cumsum  = np.cumsum(lengths)
    idx     = int(np.searchsorted(cumsum, target_tokens, side="left"))
    idx     = min(idx, len(cumsum) - 1)
    return idx + 1, int(cumsum[idx])


def slice_doc_order_into_parts(
    doc_order:  np.ndarray,
    doc_starts: np.ndarray,
    doc_ends:   np.ndarray,
    n_parts:    int,
) -> list:
    """
    Split `doc_order` into n_parts contiguous slices carrying roughly equal
    token counts.  Returns a list of (part_order, n_tokens) tuples.

    Parts are contiguous prefixes of `doc_order`, so smaller-shard nesting
    is preserved (the first part of a large shard equals the first part of
    a smaller shard when both share the same underlying global pool).
    """
    if n_parts <= 1:
        total = int((doc_ends[doc_order] - doc_starts[doc_order]).sum())
        return [(doc_order, total)]

    lengths = (doc_ends[doc_order] - doc_starts[doc_order]).astype(np.int64)
    cumsum  = np.cumsum(lengths)
    total   = int(cumsum[-1])

    target_per_part = total / n_parts
    boundaries = [0]
    for i in range(1, n_parts):
        target = int(round(target_per_part * i))
        idx    = int(np.searchsorted(cumsum, target, side="left"))
        idx    = min(idx, len(cumsum) - 1)
        if idx <= boundaries[-1]:
            idx = boundaries[-1] + 1
        boundaries.append(idx)
    boundaries.append(len(doc_order))

    parts = []
    for i in range(n_parts):
        a, b = boundaries[i], boundaries[i + 1]
        part_order  = doc_order[a:b]
        part_tokens = int(lengths[a:b].sum())
        parts.append((part_order, part_tokens))
    return parts


# ---------------------------------------------------------------------------
# Writing splits
# ---------------------------------------------------------------------------

def write_split(
    files:        list,
    doc_file_idx: np.ndarray,
    doc_starts:   np.ndarray,
    doc_ends:     np.ndarray,
    doc_order:    np.ndarray,
    out_path:     Path,
    chunk_tokens: int = 100_000_000,
) -> int:
    """
    Write the tokens for documents doc_order[0..] sequentially to out_path.

    Output is a flat raw binary uint32 file (no numpy header).

    I/O strategy: group writes by source file so each file is opened exactly
    once.  The writes into the output file are scattered (each doc lands at its
    correct output position), which is handled efficiently by the OS page cache
    on the output side.
    """
    lengths      = (doc_ends[doc_order] - doc_starts[doc_order]).astype(np.int64)
    total_tokens = int(lengths.sum())

    # Precompute every document's starting offset in the output
    out_offsets = np.zeros(len(doc_order) + 1, dtype=np.int64)
    np.cumsum(lengths, out=out_offsets[1:])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(total_tokens,))

    file_idxs   = doc_file_idx[doc_order]
    unique_fids = np.unique(file_idxs)

    for i, fid in enumerate(unique_fids):
        positions = np.where(file_idxs == fid)[0]   # positions in doc_order
        arr       = np.memmap(files[int(fid)], dtype=np.uint32, mode="r")

        for pos in positions:
            doc_idx = int(doc_order[pos])
            src_s   = int(doc_starts[doc_idx])
            n       = int(lengths[pos])
            out_s   = int(out_offsets[pos])

            copied = 0
            while copied < n:
                take = min(chunk_tokens, n - copied)
                dst[out_s + copied : out_s + copied + take] = \
                    arr[src_s + copied : src_s + copied + take]
                copied += take

        del arr

        if (i + 1) % 50 == 0 or i + 1 == len(unique_fids):
            print(f"    [{i+1}/{len(unique_fids)} source files copied]", flush=True)

    dst.flush()
    del dst
    return total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create document-sampled train/val splits from dolma2 .npy data."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root directory containing tokenized .npy files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write output split files and document index cache.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for document shuffling (default: 42).",
    )
    parser.add_argument(
        "--eos-token", type=int, default=DEFAULT_EOS_TOKEN,
        help=f"EOS / document-separator token ID "
             f"(default: {DEFAULT_EOS_TOKEN} = <|endoftext|> for dolma2-tokenizer).",
    )
    parser.add_argument(
        "--chunk-tokens", type=int, default=100_000_000,
        help="Copy chunk size in tokens during write phase (default: 100M ≈ 400 MB).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output split files.",
    )
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Force rebuild the document index even if a valid cache exists.",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated list of shard or part names to write. "
             "Accepts parent shard names (e.g. 'train_29.6B' → writes all its parts) "
             "or individual parts (e.g. 'train_29.6B.part03'). "
             "If set, all other shards are skipped even if missing. Useful for "
             "SLURM array jobs that write one part per task.",
    )
    args = parser.parse_args()

    only_shards = set()
    if args.only:
        only_shards = {s.strip() for s in args.only.split(",")}
        only_shards = {s if s.endswith(".npy") else s + ".npy" for s in only_shards}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_tokens_needed = VAL_TARGET + TRAIN_SHARDS[-1]["tokens"]

    print("=" * 72)
    print(f"  seed         : {args.seed}")
    print(f"  eos token    : {args.eos_token}")
    print(f"  tokens needed: {max_tokens_needed/1e9:.2f}B")
    print("=" * 72)

    # ── 1. Discover ALL files ────────────────────────────────────────────────
    print("\n[1/5] Discovering .npy files...")
    files = discover_files(Path(args.data_dir))
    total_avail = sum(f.stat().st_size // 4 for f in files)
    print(f"  {len(files):,} files  |  {total_avail/1e9:.1f}B tokens available")
    if total_avail < max_tokens_needed:
        raise ValueError(
            f"Not enough data: need {max_tokens_needed/1e9:.2f}B, "
            f"have {total_avail/1e9:.2f}B"
        )
    fl_hash = _file_list_hash(files)

    # ── 2. Build or load document index (covers entire corpus) ───────────────
    if (
        not args.rebuild_index
        and _cache_valid(out_dir, fl_hash, args.eos_token)
    ):
        print("\n[2/5] Loading cached document index...")
        doc_file_idx, doc_starts, doc_ends, files = load_index(out_dir)
        print(f"  {len(doc_file_idx):,} documents loaded from {out_dir/'doc_index.npz'}")
    else:
        print("\n[2/5] Building document index (scanning all files — one-time cost)...")
        print(f"  Scanning {len(files):,} files / {total_avail/1e9:.1f}B tokens ...")
        t0 = time.time()
        doc_file_idx, doc_starts, doc_ends = build_doc_index(files, args.eos_token)
        elapsed = time.time() - t0
        print(f"  Scan complete in {elapsed/60:.1f} min  |  {len(doc_file_idx):,} documents found")
        save_index(out_dir, doc_file_idx, doc_starts, doc_ends, fl_hash, files, args.eos_token)
        print(f"  Index cached to {out_dir/'doc_index.npz'}")

    n_docs         = len(doc_file_idx)
    total_indexed  = int((doc_ends - doc_starts).sum())
    avg_doc_len    = total_indexed / max(n_docs, 1)
    print(f"  {n_docs:,} documents  |  {total_indexed/1e9:.2f}B tokens  |  avg doc length {avg_doc_len:.0f} tokens")

    if total_indexed < max_tokens_needed:
        raise RuntimeError(
            f"Only {total_indexed/1e9:.2f}B tokens indexed but {max_tokens_needed/1e9:.2f}B needed. "
            "If the corpus is genuinely this small, lower the shard sizes."
        )

    # ── 3. Shuffle ALL documents globally ────────────────────────────────────
    print("\n[3/5] Shuffling all documents...")
    rng      = np.random.default_rng(args.seed)
    doc_perm = rng.permutation(n_docs)
    print(f"  Permuted {n_docs:,} documents with seed {args.seed}")

    # ── 4. Compute split boundaries ──────────────────────────────────────────
    print("\n[4/5] Computing split boundaries...")

    # Validation: first K docs in shuffled order whose tokens sum to >= VAL_TARGET
    val_n, val_actual = find_doc_cutoff(doc_perm, doc_starts, doc_ends, VAL_TARGET)
    val_order         = doc_perm[:val_n]
    print(f"  validation : {val_n:>10,} docs | {val_actual/1e9:.4f}B tokens")

    # Training: all shards share the same pool = doc_perm[val_n:]
    # Each shard is doc_perm[val_n : val_n+N_i] for increasing N_i.
    # Smaller N_i ⊆ larger N_j because they are all prefixes of the same slice.
    train_pool  = doc_perm[val_n:]
    assignments = []
    for shard in TRAIN_SHARDS:
        n, actual = find_doc_cutoff(train_pool, doc_starts, doc_ends, shard["tokens"])
        shard_order = train_pool[:n]        # nested prefix of train_pool
        assignments.append((shard, shard_order, actual))
        label = shard["name"].replace(".npy", "")
        print(f"  {label:<15}: {n:>10,} docs | {actual/1e9:.4f}B tokens")

    # ── 5. Write splits and save document indices ────────────────────────────
    total_shards = len(TRAIN_SHARDS) + 1
    print(f"\n[5/5] Writing {total_shards} splits...")

    # Save per-shard document indices as self-contained .npz files.
    # Each file stores the resolved (file_path, start, end) for every document
    # in that shard, so it remains interpretable even if the source directory
    # moves or files are added/removed.  Unique paths are deduplicated to keep
    # file sizes reasonable.
    idx_dir = out_dir / "doc_indices"
    idx_dir.mkdir(exist_ok=True)

    def _save_doc_indices(idx_path: Path, doc_order: np.ndarray):
        # Resolve file path for each document in this shard
        global_file_idxs = doc_file_idx[doc_order]          # int32, one per doc
        starts            = doc_starts[doc_order]            # int64, one per doc
        ends              = doc_ends[doc_order]              # int64, one per doc

        # Deduplicate file paths: store unique paths + a per-doc local index
        unique_global_idxs, local_file_idxs = np.unique(
            global_file_idxs, return_inverse=True
        )
        unique_paths = np.array(
            [str(files[i]) for i in unique_global_idxs], dtype=object
        )

        np.savez_compressed(
            idx_path,
            file_paths=unique_paths,       # unique source file paths (strings)
            file_idx=local_file_idxs,      # per-doc index into file_paths
            doc_starts=starts,             # token offset in that file (inclusive)
            doc_ends=ends,                 # token offset in that file (exclusive)
        )

    def _should_write(path: Path, label: str) -> bool:
        if path.exists():
            if args.overwrite:
                path.unlink()
                return True
            print(f"  [SKIP] {label} already exists. Use --overwrite to replace.")
            return False
        return True

    val_path = out_dir / "validation.npy"
    if only_shards and "validation.npy" not in only_shards:
        print(f"  [SKIP] validation.npy (not in --only filter)")
    elif _should_write(val_path, "validation.npy"):
        print(f"\n  [1/{total_shards}] validation  ({val_actual/1e9:.4f}B tokens, {val_n:,} docs)")
        idx_path = idx_dir / "validation_doc_indices.npz"
        _save_doc_indices(idx_path, val_order)
        write_split(files, doc_file_idx, doc_starts, doc_ends,
                    val_order, val_path, args.chunk_tokens)
        print(f"    ✓ {val_path}")
        print(f"    ✓ {idx_path}")

    for idx, (shard, shard_order, actual) in enumerate(assignments, start=2):
        label    = shard["name"].replace(".npy", "")
        n_parts  = num_parts_for(shard["tokens"])
        part_names = part_filenames(shard["name"], n_parts)

        # --only filter: parent shard name (with or without .npy) expands to
        # all its parts; an individual part name writes just that part.
        if only_shards:
            parent_match = shard["name"] in only_shards
            active_parts = [p for p in part_names if parent_match or p in only_shards]
            if not active_parts:
                print(f"  [SKIP] {shard['name']} (not in --only filter)")
                continue
        else:
            active_parts = part_names

        # Slice the shard's doc_order into parts (no-op if n_parts == 1).
        part_slices = slice_doc_order_into_parts(
            shard_order, doc_starts, doc_ends, n_parts
        )

        if n_parts > 1:
            print(f"\n  [{idx}/{total_shards}] {label:<15}  "
                  f"({actual/1e9:.4f}B tokens, {len(shard_order):,} docs) → {n_parts} parts")

        for part_name, (part_order, part_tokens) in zip(part_names, part_slices):
            if part_name not in active_parts:
                continue
            part_path = out_dir / part_name
            if not _should_write(part_path, part_name):
                continue

            if n_parts == 1:
                print(f"\n  [{idx}/{total_shards}] {label:<15}  "
                      f"({part_tokens/1e9:.4f}B tokens, {len(part_order):,} docs)")
            else:
                print(f"      → {part_name}  "
                      f"({part_tokens/1e9:.4f}B tokens, {len(part_order):,} docs)")

            idx_path = idx_dir / part_name.replace(".npy", "_doc_indices.npz")
            _save_doc_indices(idx_path, part_order)
            write_split(files, doc_file_idx, doc_starts, doc_ends,
                        part_order, part_path, args.chunk_tokens)
            print(f"    ✓ {part_path}")
            print(f"    ✓ {idx_path}")

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUCCESS!  Splits written:")
    if val_path.exists():
        sz = val_path.stat().st_size / 1e9
        print(f"  {'validation.npy':<26}: {val_actual:>15,} tokens  ({sz:.2f} GB)")
    for shard, shard_order, actual in assignments:
        n_parts    = num_parts_for(shard["tokens"])
        part_names = part_filenames(shard["name"], n_parts)
        for part_name in part_names:
            p  = out_dir / part_name
            if p.exists():
                sz = p.stat().st_size / 1e9
                tok = p.stat().st_size // 4
                print(f"  {part_name:<26}: {tok:>15,} tokens  ({sz:.2f} GB)")

    names = [s["name"].replace(".npy", "") for s, _, _ in assignments]
    print("\nNested structure (each ⊇ all to its right):")
    print("  " + " ⊇ ".join(reversed(names)))
    print("  Chunked shards: each parent shard's parts concatenate (in order) "
          "to form the full shard.")
    print("  All training splits are disjoint from validation.")
    print("=" * 72)


if __name__ == "__main__":
    main()
