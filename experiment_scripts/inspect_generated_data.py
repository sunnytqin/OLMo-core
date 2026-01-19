#!/usr/bin/env python3
"""
Inspect and compare generated synthetic data with real training data.

This script helps verify that generated data matches the training data format
and allows qualitative comparison of content.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect generated data and compare with training data"
    )
    parser.add_argument(
        "--generated-data",
        type=str,
        required=True,
        help="Path to generated .npy file",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Path to real training .npy file for comparison",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/chinchilla_16/30M_seed42_case3_dclm_repeat_wd0.1_lr1e-2/step4578_hf",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample documents to display",
    )
    parser.add_argument(
        "--max-tokens-per-sample",
        type=int,
        default=200,
        help="Maximum tokens to show per sample document",
    )
    parser.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Show token IDs alongside text",
    )
    return parser.parse_args()


def load_and_analyze_file(
    file_path: str,
    tokenizer,
    eos_token_id: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], dict]:
    """
    Load a numpy file and analyze its structure.

    Returns:
        - token_array: The full array of tokens
        - documents: List of (start_idx, end_idx) for each document
        - stats: Dictionary of statistics
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*80}")

    # Load array
    # Training data is stored as raw uint32 binary, not .npy format
    try:
        # Try loading as .npy format first
        token_array = np.load(file_path, mmap_mode='r')
    except (ValueError, OSError):
        # Load as raw uint32 binary (training data format)
        token_array = np.fromfile(file_path, dtype=np.uint32)

    print(f"\nFile Statistics:")
    print(f"  Shape: {token_array.shape}")
    print(f"  Dtype: {token_array.dtype}")
    print(f"  Total tokens: {len(token_array):,}")
    print(f"  File size: {Path(file_path).stat().st_size / (1024**2):.2f} MB")

    # Find EOS positions to identify documents
    eos_positions = np.where(token_array == eos_token_id)[0]
    num_eos = len(eos_positions)

    print(f"\nDocument Statistics:")
    print(f"  EOS token ID: {eos_token_id}")
    print(f"  Total EOS tokens: {num_eos:,}")
    print(f"  EOS frequency: {num_eos / len(token_array) * 100:.4f}%")

    # Create document boundaries
    # Documents are between EOS tokens: [start, eos][start, eos]...
    documents = []
    if len(eos_positions) > 0:
        # First document: from start to first EOS
        if eos_positions[0] > 0:
            documents.append((0, eos_positions[0]))

        # Middle documents: between consecutive EOS tokens
        for i in range(len(eos_positions) - 1):
            start = eos_positions[i] + 1
            end = eos_positions[i + 1]
            if start < end:
                documents.append((start, end))

        # Last document: after last EOS to end
        if eos_positions[-1] + 1 < len(token_array):
            documents.append((eos_positions[-1] + 1, len(token_array)))
    else:
        # No EOS found, treat entire array as one document
        documents.append((0, len(token_array)))

    # Calculate document length statistics
    doc_lengths = [end - start for start, end in documents]

    if doc_lengths:
        print(f"  Number of documents: {len(documents):,}")
        print(f"  Document lengths:")
        print(f"    Min: {min(doc_lengths):,} tokens")
        print(f"    Max: {max(doc_lengths):,} tokens")
        print(f"    Mean: {np.mean(doc_lengths):.1f} tokens")
        print(f"    Median: {np.median(doc_lengths):.1f} tokens")

    # Token distribution
    print(f"\nToken Distribution:")
    unique_tokens = np.unique(token_array)
    print(f"  Unique tokens: {len(unique_tokens):,}")
    print(f"  Token range: [{token_array.min()}, {token_array.max()}]")

    # Check for special tokens
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        num_pad = np.sum(token_array == pad_token_id)
        print(f"  PAD tokens (ID {pad_token_id}): {num_pad:,}")

    stats = {
        'total_tokens': len(token_array),
        'num_documents': len(documents),
        'num_eos': num_eos,
        'doc_lengths': doc_lengths,
    }

    return token_array, documents, stats


def display_samples(
    token_array: np.ndarray,
    documents: List[Tuple[int, int]],
    tokenizer,
    num_samples: int,
    max_tokens: int,
    show_token_ids: bool,
    label: str,
):
    """Display sample documents from the data."""
    print(f"\n{'='*80}")
    print(f"Sample Documents - {label}")
    print(f"{'='*80}")

    num_to_show = min(num_samples, len(documents))

    for i in range(num_to_show):
        start, end = documents[i]
        doc_tokens = token_array[start:end]

        # Truncate if too long
        truncated = False
        if len(doc_tokens) > max_tokens:
            doc_tokens = doc_tokens[:max_tokens]
            truncated = True

        print(f"\n{'-'*80}")
        print(f"Document {i+1} (tokens {start:,} to {end:,}, length: {end-start:,})")
        print(f"{'-'*80}")

        if show_token_ids:
            # Show tokens in chunks with IDs
            chunk_size = 20
            for j in range(0, len(doc_tokens), chunk_size):
                chunk = doc_tokens[j:j+chunk_size]
                ids_str = ' '.join(f'{tid:5d}' for tid in chunk)
                text_str = tokenizer.decode(chunk, skip_special_tokens=False)
                print(f"IDs:  {ids_str}")
                print(f"Text: {repr(text_str)}")
                print()
        else:
            # Just show decoded text
            decoded = tokenizer.decode(doc_tokens, skip_special_tokens=False)
            print(decoded)

        if truncated:
            print(f"\n[... truncated, showing first {max_tokens} of {end-start:,} tokens ...]")


def compare_formats(
    generated_stats: dict,
    training_stats: dict,
):
    """Compare statistics between generated and training data."""
    print(f"\n{'='*80}")
    print(f"Format Comparison")
    print(f"{'='*80}")

    print(f"\n{'Metric':<30} {'Generated':>20} {'Training':>20}")
    print(f"{'-'*72}")

    def format_num(n):
        if isinstance(n, float):
            return f"{n:,.2f}"
        return f"{n:,}"

    # Compare key metrics
    metrics = [
        ('Total tokens', 'total_tokens'),
        ('Number of documents', 'num_documents'),
        ('Number of EOS tokens', 'num_eos'),
    ]

    for label, key in metrics:
        gen_val = generated_stats.get(key, 'N/A')
        train_val = training_stats.get(key, 'N/A')
        print(f"{label:<30} {format_num(gen_val):>20} {format_num(train_val):>20}")

    # Compare document length statistics
    if 'doc_lengths' in generated_stats and 'doc_lengths' in training_stats:
        gen_lengths = generated_stats['doc_lengths']
        train_lengths = training_stats['doc_lengths']

        print(f"\n{'Document Length Stats':<30} {'Generated':>20} {'Training':>20}")
        print(f"{'-'*72}")

        comparisons = [
            ('Min length', lambda x: min(x) if x else 0),
            ('Max length', lambda x: max(x) if x else 0),
            ('Mean length', lambda x: np.mean(x) if x else 0),
            ('Median length', lambda x: np.median(x) if x else 0),
        ]

        for label, func in comparisons:
            gen_val = func(gen_lengths)
            train_val = func(train_lengths)
            print(f"{label:<30} {format_num(gen_val):>20} {format_num(train_val):>20}")


def main():
    args = parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    eos_token_id = tokenizer.eos_token_id

    print(f"Tokenizer loaded:")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  Vocab size: {len(tokenizer)}")

    # Analyze generated data
    gen_tokens, gen_docs, gen_stats = load_and_analyze_file(
        args.generated_data,
        tokenizer,
        eos_token_id,
    )

    display_samples(
        gen_tokens,
        gen_docs,
        tokenizer,
        args.num_samples,
        args.max_tokens_per_sample,
        args.show_token_ids,
        "Generated Data",
    )

    # Analyze training data if provided
    if args.training_data:
        train_tokens, train_docs, train_stats = load_and_analyze_file(
            args.training_data,
            tokenizer,
            eos_token_id,
        )

        display_samples(
            train_tokens,
            train_docs,
            tokenizer,
            args.num_samples,
            args.max_tokens_per_sample,
            args.show_token_ids,
            "Training Data",
        )

        # Compare formats
        compare_formats(gen_stats, train_stats)

    print(f"\n{'='*80}")
    print("Inspection complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
