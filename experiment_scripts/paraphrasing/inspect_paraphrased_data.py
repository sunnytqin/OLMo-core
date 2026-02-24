#!/usr/bin/env python3
"""
Inspect paraphrased .npy data and compare with original training data.

Verifies the output format is correct (EOS-delimited docs, uint32, etc.)
and displays sample paraphrased documents.

NOTE: Side-by-side comparison with the original is NOT supported because
the paraphrased .npy does not preserve a 1:1 mapping to original documents.
Documents are subsampled (every Nth) and some are skipped (too long), so
there is no way to reliably align paraphrased doc i to original doc j.
Use --original to compare aggregate statistics (token counts, doc lengths).
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tokenizers import Tokenizer as DolmaTokenizer

EOS_TOKEN_ID = 100257


def load_dolma2_tokenizer():
    return DolmaTokenizer.from_pretrained("allenai/dolma2-tokenizer")


def load_and_analyze_file(file_path, tokenizer):
    """Load a .npy file and analyze its structure."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*80}")

    arr = np.memmap(file_path, dtype=np.uint32, mode="r")

    print(f"\nFile Statistics:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Total tokens: {len(arr):,}")
    print(f"  File size: {Path(file_path).stat().st_size / (1024**2):.2f} MB")

    # Find EOS positions
    eos_positions = np.where(arr == EOS_TOKEN_ID)[0]

    print(f"\nDocument Statistics:")
    print(f"  EOS token ID: {EOS_TOKEN_ID}")
    print(f"  Total EOS tokens: {len(eos_positions):,}")
    print(f"  EOS frequency: {len(eos_positions) / len(arr) * 100:.4f}%")

    # Extract document boundaries
    documents = []
    if len(eos_positions) > 0:
        if eos_positions[0] > 0:
            documents.append((0, int(eos_positions[0])))
        for i in range(len(eos_positions) - 1):
            start = int(eos_positions[i]) + 1
            end = int(eos_positions[i + 1])
            if start < end:
                documents.append((start, end))
        if int(eos_positions[-1]) + 1 < len(arr):
            documents.append((int(eos_positions[-1]) + 1, len(arr)))

    doc_lengths = [end - start for start, end in documents]

    if doc_lengths:
        print(f"  Number of documents: {len(documents):,}")
        print(f"  Document lengths:")
        print(f"    Min: {min(doc_lengths):,} tokens")
        print(f"    Max: {max(doc_lengths):,} tokens")
        print(f"    Mean: {np.mean(doc_lengths):.1f} tokens")
        print(f"    Median: {np.median(doc_lengths):.1f} tokens")

    # Token distribution
    unique_tokens = np.unique(arr)
    print(f"\nToken Distribution:")
    print(f"  Unique tokens: {len(unique_tokens):,}")
    print(f"  Token range: [{arr.min()}, {arr.max()}]")

    stats = {
        "total_tokens": len(arr),
        "num_documents": len(documents),
        "num_eos": len(eos_positions),
        "doc_lengths": doc_lengths,
    }

    return arr, documents, stats


def display_samples(arr, documents, tokenizer, num_samples, max_chars, label):
    """Display sample documents."""
    print(f"\n{'='*80}")
    print(f"Sample Documents - {label}")
    print(f"{'='*80}")

    num_to_show = min(num_samples, len(documents))
    for i in range(num_to_show):
        start, end = documents[i]
        doc_tokens = arr[start:end].tolist()
        text = tokenizer.decode(doc_tokens, skip_special_tokens=False)

        print(f"\n{'-'*80}")
        print(f"Document {i+1} ({end - start:,} tokens, {len(text):,} chars)")
        print(f"{'-'*80}")
        if max_chars > 0 and len(text) > max_chars:
            print(text[:max_chars] + "...")
        else:
            print(text)


def compare_formats(gen_stats, train_stats):
    """Compare statistics between paraphrased and original data."""
    print(f"\n{'='*80}")
    print(f"Format Comparison")
    print(f"{'='*80}")

    def fmt(n):
        if isinstance(n, float):
            return f"{n:,.2f}"
        return f"{n:,}"

    print(f"\n{'Metric':<30} {'Paraphrased':>20} {'Original':>20}")
    print(f"{'-'*72}")

    for label, key in [
        ("Total tokens", "total_tokens"),
        ("Number of documents", "num_documents"),
        ("Number of EOS tokens", "num_eos"),
    ]:
        print(f"{label:<30} {fmt(gen_stats[key]):>20} {fmt(train_stats[key]):>20}")

    gen_lengths = gen_stats["doc_lengths"]
    train_lengths = train_stats["doc_lengths"]

    print(f"\n{'Document Length Stats':<30} {'Paraphrased':>20} {'Original':>20}")
    print(f"{'-'*72}")

    for label, func in [
        ("Min length", lambda x: min(x) if x else 0),
        ("Max length", lambda x: max(x) if x else 0),
        ("Mean length", lambda x: np.mean(x) if x else 0),
        ("Median length", lambda x: np.median(x) if x else 0),
    ]:
        print(f"{label:<30} {fmt(func(gen_lengths)):>20} {fmt(func(train_lengths)):>20}")


def main():
    parser = argparse.ArgumentParser(description="Inspect paraphrased data and compare with original")
    parser.add_argument("--paraphrased", type=str, required=True, help="Path to paraphrased .npy file")
    parser.add_argument("--original", type=str, default=None, help="Path to original .npy file for comparison")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sample documents to display")
    parser.add_argument("--max-chars", type=int, default=1000, help="Max characters to display per document (0 for full)")
    args = parser.parse_args()

    print("Loading dolma2 tokenizer...")
    tokenizer = load_dolma2_tokenizer()

    # Analyze paraphrased data
    para_arr, para_docs, para_stats = load_and_analyze_file(args.paraphrased, tokenizer)
    display_samples(para_arr, para_docs, tokenizer, args.num_samples, args.max_chars, "Paraphrased Data")

    # Compare aggregate stats with original if provided
    if args.original:
        orig_arr, orig_docs, orig_stats = load_and_analyze_file(args.original, tokenizer)
        compare_formats(para_stats, orig_stats)

    print(f"\n{'='*80}")
    print("Inspection complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
