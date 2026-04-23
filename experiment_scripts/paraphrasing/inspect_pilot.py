#!/usr/bin/env python3
"""
Inspect a pilot paraphrase run.

Reads the input .npy + the shard_XXXX.jsonl checkpoint produced by
paraphrase_shard.py, and reports:

  - Length-preservation ratio (output_tokens / input_tokens) — p10/p50/p90
  - Fraction of outputs that hit max_tokens (potential truncation)
  - Fraction of outputs that are near-duplicates of input (copy, not rephrase)
  - A handful of random side-by-side samples for manual quality eyeball
  - Aggregate token counts (for throughput back-calculation)

Usage mirrors paraphrase_shard.py's sharding args so the "same N docs" are
reconstructed on the input side.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer as DolmaTokenizer

EOS_TOKEN_ID = 100257


def load_tokenizer():
    return DolmaTokenizer.from_pretrained("allenai/dolma2-tokenizer")


def extract_shard_documents(arr, shard_id, num_shards, subsample):
    """Replicate paraphrase_shard.extract_shard_documents — returns list of token lists."""
    eos_positions = np.where(arr == EOS_TOKEN_ID)[0]
    total_docs = len(eos_positions)
    selected_indices = list(range(0, total_docs, subsample))
    num_selected = len(selected_indices)
    docs_per_shard = num_selected // num_shards
    shard_start = shard_id * docs_per_shard
    shard_end = shard_start + docs_per_shard if shard_id < num_shards - 1 else num_selected

    docs = []
    for s in range(shard_start, shard_end):
        i = selected_indices[s]
        doc_start = 0 if i == 0 else int(eos_positions[i - 1]) + 1
        doc_end = int(eos_positions[i])
        docs.append(arr[doc_start:doc_end].tolist())
    return docs


def token_ngram_set(tokens, n=8):
    """Set of n-gram tuples from a token list — used to estimate copy overlap."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Original .npy (same as paraphrase_shard --input)")
    p.add_argument("--checkpoint", required=True, help="shard_XXXX.jsonl from paraphrase run")
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=10, help="Random side-by-side samples to print")
    p.add_argument("--max-tokens-check", type=int, default=16384, help="Max output tokens used during generation (for truncation check)")
    p.add_argument("--ngram", type=int, default=8, help="N-gram size for copy-overlap detection")
    args = p.parse_args()

    tok = load_tokenizer()

    print(f"Loading input: {args.input}")
    arr = np.memmap(args.input, dtype=np.uint32, mode="r")
    print(f"Reconstructing shard documents (shard {args.shard_id}/{args.num_shards}, subsample={args.subsample})...")
    input_docs = extract_shard_documents(arr, args.shard_id, args.num_shards, args.subsample)
    print(f"  {len(input_docs):,} input docs")

    print(f"Loading paraphrases: {args.checkpoint}")
    paraphrases = {}
    with open(args.checkpoint) as f:
        for line in f:
            e = json.loads(line)
            paraphrases[e["idx"]] = e["text"]
    print(f"  {len(paraphrases):,} paraphrased entries")

    aligned = []
    for i, doc_tokens in enumerate(input_docs):
        if i not in paraphrases:
            continue
        para_text = paraphrases[i]
        if not para_text or not para_text.strip():
            aligned.append((i, doc_tokens, para_text, None))
            continue
        para_tokens = tok.encode(para_text.strip()).ids
        aligned.append((i, doc_tokens, para_text, para_tokens))

    n_aligned = len(aligned)
    n_empty = sum(1 for _, _, t, toks in aligned if toks is None)
    n_nonempty = n_aligned - n_empty
    print(f"  aligned: {n_aligned:,} ({n_empty} empty outputs)")

    # Length ratio
    ratios = []
    input_tok_counts = []
    output_tok_counts = []
    near_max = 0
    copy_heavy = 0
    zero_overlap = 0
    for i, in_toks, para_text, out_toks in aligned:
        if out_toks is None:
            continue
        n_in = len(in_toks)
        n_out = len(out_toks)
        input_tok_counts.append(n_in)
        output_tok_counts.append(n_out)
        if n_in > 0:
            ratios.append(n_out / n_in)
        if n_out >= args.max_tokens_check - 5:
            near_max += 1

        # Copy-overlap heuristic: what fraction of output n-grams are literally in the input?
        in_set = token_ngram_set(in_toks, args.ngram)
        out_set = token_ngram_set(out_toks, args.ngram)
        if out_set:
            overlap = len(in_set & out_set) / len(out_set)
            if overlap > 0.5:
                copy_heavy += 1
            if overlap == 0 and len(out_set) > 5:
                zero_overlap += 1

    ratios = np.array(ratios)
    print("\n=== Length preservation (output_tokens / input_tokens) ===")
    if len(ratios):
        print(f"  n       : {len(ratios):,}")
        print(f"  mean    : {ratios.mean():.3f}")
        print(f"  p10     : {np.percentile(ratios, 10):.3f}")
        print(f"  p50     : {np.percentile(ratios, 50):.3f}")
        print(f"  p90     : {np.percentile(ratios, 90):.3f}")
        print(f"  max     : {ratios.max():.3f}")
        print(f"  fraction with ratio > 1.5: {(ratios > 1.5).mean():.2%}")
        print(f"  fraction with ratio < 0.5: {(ratios < 0.5).mean():.2%}")

    print(f"\n=== Token counts ===")
    print(f"  total input tokens  : {sum(input_tok_counts):,}")
    print(f"  total output tokens : {sum(output_tok_counts):,}")
    print(f"  paraphrase ratio    : {sum(output_tok_counts)/max(1,sum(input_tok_counts)):.3f}")

    print(f"\n=== Red flags ===")
    print(f"  empty outputs        : {n_empty} / {n_aligned}")
    print(f"  near max_tokens (potential truncation): {near_max} / {n_nonempty}")
    print(f"  copy-heavy (>{50}% {args.ngram}-gram overlap with input): {copy_heavy} / {n_nonempty}")
    print(f"  zero n-gram overlap  : {zero_overlap} / {n_nonempty}  (suspicious if near 0 or high — see samples)")

    # Random samples
    print(f"\n=== {args.num_samples} random side-by-side samples ===")
    nonempty = [a for a in aligned if a[3] is not None]
    if not nonempty:
        print("  (no non-empty outputs to sample)")
        return
    rng = np.random.default_rng(0)
    sample_idxs = rng.choice(len(nonempty), size=min(args.num_samples, len(nonempty)), replace=False)
    for si, s in enumerate(sample_idxs):
        i, in_toks, para_text, out_toks = nonempty[s]
        orig_text = tok.decode(in_toks, skip_special_tokens=False)
        n_in = len(in_toks)
        n_out = len(out_toks)
        ratio = n_out / max(1, n_in)
        print(f"\n--- Sample {si+1}  (doc idx {i}  in={n_in} tok  out={n_out} tok  ratio={ratio:.2f}) ---")
        print("ORIGINAL:")
        print(orig_text[:1200] + ("..." if len(orig_text) > 1200 else ""))
        print("\nPARAPHRASE:")
        print(para_text[:1200] + ("..." if len(para_text) > 1200 else ""))


if __name__ == "__main__":
    main()
