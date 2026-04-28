#!/usr/bin/env python3
"""
Inspect a pilot paraphrase run.

Reads the input .npy + the shard_XXXX.jsonl checkpoint produced by
paraphrase_shard.py, and reports:

  - Prompt-style distribution (for mixed-mode runs)
  - Length-preservation ratio per prompt style (p10/p50/p90)
  - Fraction of outputs that hit max_tokens (potential truncation)
  - Fraction of outputs that are near-duplicates of input (copy, not rephrase)
  - Fraction of model refusals ("I cannot", "The document does not contain", ...)
  - Per-style random side-by-side samples for manual quality eyeball
  - Aggregate token counts (for throughput back-calculation)

Usage mirrors paraphrase_shard.py's sharding args so the "same N docs" are
reconstructed on the input side.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer as DolmaTokenizer

EOS_TOKEN_ID = 100257

# Case-insensitive opener patterns that indicate the model refused the task
# rather than producing the requested format (e.g., math on non-numeric doc).
REFUSAL_PATTERNS = re.compile(
    r"^\s*(the document (does not|doesn't) contain"
    r"|i (cannot|can't|am unable|'m unable)"
    r"|there (are|is) no (numerical|numeric|relevant)"
    r"|unfortunately"
    r"|there's no|there is nothing"
    r"|sorry,?\s*(i|but))",
    re.IGNORECASE,
)


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


def summarize_ratios(label, ratios):
    """Print mean/p10/p50/p90 + tail fractions for a set of output/input ratios."""
    if not len(ratios):
        print(f"  {label:<10}: (no data)")
        return
    r = np.array(ratios)
    print(
        f"  {label:<10}  n={len(r):>4}  mean={r.mean():.2f}  "
        f"p10={np.percentile(r,10):.2f}  p50={np.percentile(r,50):.2f}  "
        f"p90={np.percentile(r,90):.2f}  "
        f"<0.5×: {(r<0.5).mean():>5.1%}  >1.5×: {(r>1.5).mean():>5.1%}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Original .npy (same as paraphrase_shard --input)")
    p.add_argument("--checkpoint", required=True, help="shard_XXXX.jsonl from paraphrase run")
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--samples-per-style", type=int, default=3, help="Side-by-side samples to print per prompt style")
    p.add_argument("--max-tokens-check", type=int, default=8192, help="Max output tokens used during generation (for truncation check)")
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
            paraphrases[e["idx"]] = (e["text"], e.get("prompt_style"))  # style may be None (legacy)
    print(f"  {len(paraphrases):,} paraphrased entries")

    # Align input docs with paraphrased outputs
    aligned = []  # list of (idx, in_toks, para_text, out_toks, style)
    for i, doc_tokens in enumerate(input_docs):
        if i not in paraphrases:
            continue
        para_text, style = paraphrases[i]
        if not para_text or not para_text.strip():
            aligned.append((i, doc_tokens, para_text, None, style))
            continue
        para_tokens = tok.encode(para_text.strip()).ids
        aligned.append((i, doc_tokens, para_text, para_tokens, style))

    n_aligned = len(aligned)
    n_empty = sum(1 for _, _, _, toks, _ in aligned if toks is None)
    n_nonempty = n_aligned - n_empty
    print(f"  aligned: {n_aligned:,} ({n_empty} empty outputs)")

    # Prompt-style distribution
    style_counts = Counter(a[4] for a in aligned)
    print("\n=== Prompt distribution ===")
    for style, c in sorted(style_counts.items(), key=lambda x: (x[0] is None, x[0] or "")):
        label = style if style is not None else "(legacy: no style field)"
        print(f"  {label:<30}: {c:>5}  ({100*c/max(1,n_aligned):>5.1f}%)")

    # Per-style stats
    per_style_ratios = defaultdict(list)
    all_ratios = []
    input_tok_counts = []
    output_tok_counts = []
    near_max = 0
    copy_heavy = 0
    zero_overlap = 0
    refusals = Counter()
    per_style_refusals = Counter()

    for i, in_toks, para_text, out_toks, style in aligned:
        if out_toks is None:
            continue
        n_in = len(in_toks)
        n_out = len(out_toks)
        input_tok_counts.append(n_in)
        output_tok_counts.append(n_out)
        if n_in > 0:
            r = n_out / n_in
            all_ratios.append(r)
            per_style_ratios[style].append(r)
        if n_out >= args.max_tokens_check - 5:
            near_max += 1

        # Copy-overlap heuristic
        in_set = token_ngram_set(in_toks, args.ngram)
        out_set = token_ngram_set(out_toks, args.ngram)
        if out_set:
            overlap = len(in_set & out_set) / len(out_set)
            if overlap > 0.5:
                copy_heavy += 1
            if overlap == 0 and len(out_set) > 5:
                zero_overlap += 1

        # Refusal detector (check first 200 chars)
        if REFUSAL_PATTERNS.search(para_text[:200]):
            refusals[style] += 1

    print("\n=== Length preservation (output_tokens / input_tokens), by prompt style ===")
    summarize_ratios("ALL", all_ratios)
    for style in sorted(per_style_ratios.keys(), key=lambda s: (s is None, s or "")):
        label = style if style is not None else "(legacy)"
        summarize_ratios(label, per_style_ratios[style])

    print(f"\n=== Token counts ===")
    print(f"  total input tokens  : {sum(input_tok_counts):,}")
    print(f"  total output tokens : {sum(output_tok_counts):,}")
    print(f"  paraphrase ratio    : {sum(output_tok_counts)/max(1,sum(input_tok_counts)):.3f}")

    print(f"\n=== Red flags ===")
    print(f"  empty outputs        : {n_empty} / {n_aligned}")
    print(f"  near max_tokens      : {near_max} / {n_nonempty}  (potential truncation)")
    print(f"  copy-heavy (>{50}% {args.ngram}-gram overlap): {copy_heavy} / {n_nonempty}")
    print(f"  zero n-gram overlap  : {zero_overlap} / {n_nonempty}")
    total_refusals = sum(refusals.values())
    print(f"  refusals             : {total_refusals} / {n_nonempty}  ({100*total_refusals/max(1,n_nonempty):.1f}%)")
    if total_refusals:
        for style, c in sorted(refusals.items(), key=lambda x: (x[0] is None, x[0] or "")):
            print(f"    {style}: {c}")

    # Per-style random samples
    nonempty_by_style = defaultdict(list)
    for a in aligned:
        if a[3] is not None:
            nonempty_by_style[a[4]].append(a)

    rng = np.random.default_rng(0)
    for style in sorted(nonempty_by_style.keys(), key=lambda s: (s is None, s or "")):
        samples = nonempty_by_style[style]
        if not samples:
            continue
        k = min(args.samples_per_style, len(samples))
        picks = rng.choice(len(samples), size=k, replace=False)
        print(f"\n=== {k} random samples for prompt style: {style!r} ===")
        for si, s in enumerate(picks):
            i, in_toks, para_text, out_toks, _ = samples[s]
            orig_text = tok.decode(in_toks, skip_special_tokens=False)
            n_in = len(in_toks)
            n_out = len(out_toks)
            ratio = n_out / max(1, n_in)
            print(f"\n--- [{style}] Sample {si+1}  (doc idx {i}  in={n_in} tok  out={n_out} tok  ratio={ratio:.2f}) ---")
            print("ORIGINAL:")
            print(orig_text[:1000] + ("..." if len(orig_text) > 1000 else ""))
            print("\nPARAPHRASE:")
            print(para_text[:1000] + ("..." if len(para_text) > 1000 else ""))


if __name__ == "__main__":
    main()
