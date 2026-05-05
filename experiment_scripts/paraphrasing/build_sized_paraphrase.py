#!/usr/bin/env python3
"""
Build size-aligned paraphrase shards from V2 SmolLM2 mixed paraphrase output.

For a target dolma shard like train_2.4B (with N source docs), this script
produces paraphrased/sized_smollm2_mixed/train_2.4B_seedK.npy = the paraphrases
of all source docs with idx < N for seed K.

Doc-alignment via the JSONL `idx` field
---------------------------------------
The .jsonl files preserve a strict 1:1 mapping between source docs and entries
(every shard has 152,085 jsonl entries with idx 0..152,084 in order, identical
across all 8 seeds). However the .npy files only contain tokens for docs whose
paraphrase had non-empty text — refused/empty paraphrases are dropped from the
.npy entirely. Refusal counts vary across seeds.

So "first M EOS-terminated chunks of npy" is NOT a safe slicing scheme — when a
shard has refusals, M successful paraphrases can correspond to source idx range
beyond M. We instead read the jsonl to find which source idxs survived, cache
those per (seed, shard), and slice the npy to keep only docs whose source idx
< target N. This guarantees `D'` strictly contains paraphrases of `D`'s docs.

Consequence: D' doc count varies slightly across seeds (different refusal
patterns). Total tokens are computed from on-disk file sizes by the training
script, so this asymmetry is automatic.

Usage:
    python build_sized_paraphrase.py \\
        --shards train_0.03B,train_2.4B \\
        --seeds 1,2

Output side-effects:
- {OUTPUT_DIR}/{shard}_seed{N}.npy
- {OUTPUT_DIR}/manifest_seed{N}.json   (per-seed; avoids parallel-write race)
- {OUTPUT_DIR}/seed{N}_shard{i:04d}_idxs.npy   (cache of successful source idxs)
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

EOS_TOKEN_ID = 100257
DTYPE = np.uint32
ITEMSIZE = np.dtype(DTYPE).itemsize  # = 4

# strip_preamble + PREAMBLE_OPENERS are copied from paraphrase_shard.py so we
# can reproduce the EXACT tokenization the writer used (without importing
# paraphrase_shard.py, which pulls in vLLM at module load time).
PREAMBLE_OPENERS = (
    "Here is", "Here's", "Here follows", "Below is",
    "The following is", "I have", "I'll", "Let me",
    "Sure,", "Certainly,", "Of course,",
)


def strip_preamble(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    first_nl = text.find("\n")
    if first_nl < 0:
        return text
    first_line = text[:first_nl]
    if not any(first_line.startswith(p) for p in PREAMBLE_OPENERS):
        return text
    sep = text.find("\n---\n")
    if 0 < sep < 600:
        return text[sep + len("\n---\n"):].lstrip()
    sep = text.find("\n---")
    if 0 < sep < 600 and text[sep:sep+8] in ("\n---\n\n", "\n---\n", "\n----\n"):
        return text[sep:].lstrip("-\n ")
    if first_line.rstrip().endswith(":"):
        return text[first_nl + 1:].lstrip()
    para_break = text.find("\n\n")
    if 0 < para_break < 500:
        return text[para_break + 2:].lstrip()
    return text


_DOLMA2_TOK = None


def get_dolma2_tokenizer():
    """Lazy-load the dolma2 fast tokenizer used by paraphrase_shard.tokenize_to_npy."""
    global _DOLMA2_TOK
    if _DOLMA2_TOK is None:
        os.environ.setdefault("HF_HOME", "/n/netscratch/barak_lab/Lab/sqin/cache")
        from tokenizers import Tokenizer as DolmaTokenizer
        _DOLMA2_TOK = DolmaTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    return _DOLMA2_TOK

DATA_ROOT = Path(
    "/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/"
    "resharded/allenai/dolma2-tokenizer"
)
SHARD_MANIFEST = DATA_ROOT / "shard_manifest.json"
PARAPHRASE_ROOT = DATA_ROOT / "paraphrased"
SEED_DIR_TEMPLATE = "train_7.4B_smollm2_mixed_seed{seed}"
OUTPUT_DIR = PARAPHRASE_ROOT / "sized_smollm2_mixed"

# 32 shards per seed. Per the README, jsonl has 152,085 entries in shards 0..30
# and 152,097 in shard 31, totaling 4,866,732 source docs.
SHARDS_PER_SEED = 32
DOCS_PER_FULL_SHARD = 152_085
PARAPHRASE_MAX_IDX_EXCLUSIVE = 4_866_732  # inherited bound from train_7.4B docs


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def count_eos(path: Path) -> int:
    arr = np.memmap(path, dtype=DTYPE, mode="r")
    return int(np.count_nonzero(arr == EOS_TOKEN_ID))


def _idxs_cache_path(seed: int, shard_idx: int) -> Path:
    return OUTPUT_DIR / f"seed{seed}_shard{shard_idx:04d}_idxs.npz"


def _manifest_path(seed: int) -> Path:
    return OUTPUT_DIR / f"manifest_seed{seed}.json"


def get_doc_offsets(seed: int, shard_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (idx, end_offset) arrays describing every successfully-paraphrased
    doc in seed{N}/shard_{idx:04d}.npy:
      idx[k]        = source idx of the k-th successful doc (jsonl ordering)
      end_offset[k] = token offset in the .npy just past the k-th doc's
                      terminating EOS (i.e. bytes [end_offset[k-1], end_offset[k])
                      are the k-th doc's tokens followed by its EOS terminator)

    Two paths:
      * Fast path: jsonl_nonempty_count == npy_eos_count. Then npy EOS positions
        ARE the doc terminators, no re-tokenization needed.
      * Slow path: counts disagree (paraphrase model emitted EOS-like text mid
        doc, producing extra EOS tokens in npy). Re-tokenize each non-empty
        jsonl text using the dolma2 tokenizer to recover the true byte offset
        of each doc's terminator.

    Cached on disk per (seed, shard).
    """
    cache = _idxs_cache_path(seed, shard_idx)
    if cache.exists():
        with np.load(cache) as d:
            return d["idx"].copy(), d["end_offset"].copy()

    seed_dir = PARAPHRASE_ROOT / SEED_DIR_TEMPLATE.format(seed=seed)
    jsonl = seed_dir / f"shard_{shard_idx:04d}.jsonl"
    npy = seed_dir / f"shard_{shard_idx:04d}.npy"
    if not jsonl.exists():
        raise FileNotFoundError(f"Missing paraphrase jsonl: {jsonl}")
    if not npy.exists():
        raise FileNotFoundError(f"Missing paraphrase npy: {npy}")

    # The .jsonl `idx` field is LOCAL to each shard (each shard has idx 0..152084,
    # or 0..152096 for shard 31). Convert to GLOBAL source idx via the README's
    # invariant: global = shard_idx * 152085 + local_idx (covers all 32 shards
    # since shard 31 just extends past 31*152085).
    shard_global_offset = shard_idx * DOCS_PER_FULL_SHARD
    successful_idxs: List[int] = []
    successful_texts: List[str] = []
    with open(jsonl) as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "") or ""
            if text.strip():
                successful_idxs.append(shard_global_offset + int(rec["idx"]))
                successful_texts.append(text)

    idx_array = np.asarray(successful_idxs, dtype=np.uint32)
    if idx_array.size > 1 and not np.all(np.diff(idx_array) > 0):
        raise RuntimeError(
            f"Successful idxs in {seed_dir.name}/shard_{shard_idx:04d} are not "
            f"strictly ascending — alignment assumption violated."
        )

    arr = np.memmap(npy, dtype=DTYPE, mode="r")
    n_eos = int(np.count_nonzero(arr == EOS_TOKEN_ID))
    n_jsonl = idx_array.size

    if n_eos == n_jsonl:
        eos_positions = np.where(arr == EOS_TOKEN_ID)[0]
        end_offsets = (eos_positions + 1).astype(np.uint64)
    else:
        # Slow path: re-tokenize to find true doc boundaries.
        if n_eos < n_jsonl:
            raise RuntimeError(
                f"{seed_dir.name}/shard_{shard_idx:04d}: npy has {n_eos} EOS but "
                f"jsonl has {n_jsonl} non-empty entries — npy missing docs, cannot recover."
            )
        print(
            f"  [{seed_dir.name}/shard_{shard_idx:04d}] embedded-EOS detected "
            f"(npy_eos={n_eos:,}, jsonl_nonempty={n_jsonl:,}); re-tokenizing...",
            flush=True,
        )
        tok = get_dolma2_tokenizer()
        end_offsets = np.empty(n_jsonl, dtype=np.uint64)
        pos = 0
        n_total = len(arr)
        # Use batch encoding for speed.
        BATCH = 512
        cursor = 0
        for start in range(0, n_jsonl, BATCH):
            batch_texts = [strip_preamble(t) for t in successful_texts[start:start + BATCH]]
            encs = tok.encode_batch(batch_texts)
            for enc in encs:
                doc_len = len(enc.ids)
                end = pos + doc_len + 1  # +1 for trailing EOS
                if end > n_total:
                    raise RuntimeError(
                        f"npy too short while re-tokenizing "
                        f"{seed_dir.name}/shard_{shard_idx:04d} at doc {cursor}: "
                        f"need {end}, have {n_total}"
                    )
                if int(arr[end - 1]) != EOS_TOKEN_ID:
                    raise RuntimeError(
                        f"Re-tokenization byte-mismatch at "
                        f"{seed_dir.name}/shard_{shard_idx:04d} doc {cursor} "
                        f"(idx {int(idx_array[cursor])}): expected EOS at npy[{end-1}], "
                        f"got {int(arr[end - 1])}."
                    )
                end_offsets[cursor] = end
                pos = end
                cursor += 1
        if pos != n_total:
            raise RuntimeError(
                f"Re-tokenization consumed {pos:,} of {n_total:,} tokens in "
                f"{seed_dir.name}/shard_{shard_idx:04d} — trailing tokens unaccounted for."
            )
    del arr

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, idx=idx_array, end_offset=end_offsets)
    return idx_array, end_offsets


def plan_extracts(
    seed: int, target_num_docs: int
) -> List[Tuple[int, int, int]]:
    """
    Returns list of (shard_idx, n_take, end_byte) where n_take is the number
    of successful paraphrase docs from this shard whose source idx is < N, and
    end_byte is the npy byte offset immediately past their last EOS terminator.
    """
    if target_num_docs > PARAPHRASE_MAX_IDX_EXCLUSIVE:
        raise ValueError(
            f"target_num_docs={target_num_docs:,} exceeds paraphrase coverage "
            f"({PARAPHRASE_MAX_IDX_EXCLUSIVE:,} source docs)."
        )
    out: List[Tuple[int, int, int]] = []
    for shard_idx in range(SHARDS_PER_SEED):
        if shard_idx * DOCS_PER_FULL_SHARD >= target_num_docs:
            break
        idxs, end_offsets = get_doc_offsets(seed, shard_idx)
        n_take = int(np.searchsorted(idxs, target_num_docs, side="left"))
        if n_take > 0:
            out.append((shard_idx, n_take, int(end_offsets[n_take - 1])))
    return out


def build_one(
    target_shard: str,
    seed: int,
    target_num_docs: int,
    overwrite: bool,
) -> dict:
    out_path = OUTPUT_DIR / f"{target_shard}_seed{seed}.npy"
    if out_path.exists() and not overwrite:
        actual_tokens = out_path.stat().st_size // ITEMSIZE
        actual_docs = count_eos(out_path)
        print(
            f"  [SKIP] {out_path.name} exists ({actual_tokens:,} tokens, "
            f"{actual_docs:,} docs). Use --overwrite to replace.",
            flush=True,
        )
        return {"num_tokens": actual_tokens, "num_docs": actual_docs, "skipped": True}

    seed_dir = PARAPHRASE_ROOT / SEED_DIR_TEMPLATE.format(seed=seed)
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    extracts = plan_extracts(seed, target_num_docs)
    if not extracts:
        raise ValueError(
            f"No paraphrase docs match target_num_docs={target_num_docs} for seed={seed}."
        )

    total_tokens = sum(end_byte for _, _, end_byte in extracts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = np.memmap(out_path, dtype=DTYPE, mode="w+", shape=(total_tokens,))
    offset = 0
    actual_docs = 0
    for shard_idx, n_take, end_byte in extracts:
        npy_path = seed_dir / f"shard_{shard_idx:04d}.npy"
        src = np.memmap(npy_path, dtype=DTYPE, mode="r")
        out[offset : offset + end_byte] = src[:end_byte]
        offset += end_byte
        actual_docs += n_take
        del src
    assert offset == total_tokens, (offset, total_tokens)
    out.flush()
    del out

    return {"num_tokens": total_tokens, "num_docs": actual_docs, "skipped": False}


def update_manifest(seed: int, target_shard: str, info: dict) -> None:
    """Per-seed manifest — each task only touches its own file (no race)."""
    path = _manifest_path(seed)
    manifest: dict = {}
    if path.exists():
        try:
            with open(path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            manifest = {}
    manifest[target_shard] = {
        "num_tokens": info["num_tokens"],
        "num_docs": info["num_docs"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build size-aligned paraphrase shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--shards",
        required=True,
        help="Comma-separated target dolma shard basenames (no .npy).",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated seed numbers in 1..16.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild even if the output file exists.",
    )
    args = parser.parse_args()

    target_shards = [s.strip() for s in args.shards.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for s in seeds:
        if not 1 <= s <= 16:
            raise ValueError(f"Seed must be in 1..16, got {s}")

    if not SHARD_MANIFEST.exists():
        raise FileNotFoundError(f"shard_manifest.json not found at {SHARD_MANIFEST}")
    with open(SHARD_MANIFEST) as f:
        manifest = json.load(f)
    shards_meta = manifest.get("shards", manifest)

    targets = {}
    for shard in target_shards:
        key = f"{shard}.npy"
        if key not in shards_meta:
            raise KeyError(
                f"{key!r} not in shard_manifest.json. "
                f"Available (first 10): {list(shards_meta)[:10]}"
            )
        n_docs = shards_meta[key]["num_docs"]
        if n_docs > PARAPHRASE_MAX_IDX_EXCLUSIVE:
            raise ValueError(
                f"Shard {shard} has {n_docs:,} docs, exceeds paraphrase coverage "
                f"({PARAPHRASE_MAX_IDX_EXCLUSIVE:,})."
            )
        targets[shard] = n_docs

    print("=" * 72, flush=True)
    print(f"Output dir : {OUTPUT_DIR}", flush=True)
    print(
        f"Targets    : {len(target_shards)} shards x {len(seeds)} seeds = "
        f"{len(target_shards) * len(seeds)} files",
        flush=True,
    )
    for shard, n_docs in targets.items():
        print(f"  {shard}: {n_docs:,} source docs", flush=True)
    print(f"Seeds      : {seeds}", flush=True)
    print("=" * 72, flush=True)

    t0 = time.time()
    for shard in target_shards:
        target_num_docs = targets[shard]
        for seed in seeds:
            t_start = time.time()
            print(
                f"\nBuilding {shard}_seed{seed}.npy (source idxs [0, {target_num_docs:,}))...",
                flush=True,
            )
            info = build_one(shard, seed, target_num_docs, args.overwrite)
            update_manifest(seed, shard, info)
            elapsed = time.time() - t_start
            tag = " (skipped)" if info["skipped"] else ""
            n_missing = target_num_docs - info["num_docs"]
            miss_tag = f", missing {n_missing} (refusals)" if n_missing > 0 else ""
            print(
                f"  -> {info['num_tokens']:,} tokens, {info['num_docs']:,} docs"
                f"{miss_tag} in {elapsed:.1f}s{tag}",
                flush=True,
            )

    elapsed_total = time.time() - t0
    print("\n" + "=" * 72, flush=True)
    print(f"Done in {elapsed_total/60:.2f} min", flush=True)
    for seed in seeds:
        print(f"  manifest: {_manifest_path(seed)}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
