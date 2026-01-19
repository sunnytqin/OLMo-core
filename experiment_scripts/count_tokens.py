#!/usr/bin/env python3
"""
Script to count total tokens in the DCLM training dataset.
"""

import argparse
import numpy as np
from pathlib import Path
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.io import is_url, normalize_path
import sys

def count_tokens_in_file(file_path):
    """Count tokens in a single numpy file."""
    try:
        if is_url(file_path):
            # For remote files, estimate from file size
            from olmo_core.io import get_file_size
            size = get_file_size(file_path)
            # Assume uint32 (4 bytes per token)
            estimated_tokens = size // 4
            print(f"  {Path(file_path).name}: ~{estimated_tokens:,} tokens (remote, {size:,} bytes)")
            return estimated_tokens, size
        else:
            # For local files, memory-map as uint32 to count tokens
            file_size = Path(file_path).stat().st_size
            # Memory-map as uint32 (each token is a 4-byte integer)
            data = np.memmap(file_path, dtype=np.uint32, mode='r')
            token_count = data.size  # This is the total number of tokens!
            print(f"  {Path(file_path).name}: {token_count:,} tokens ({file_size:,} bytes)")
            del data  # Free the memmap
            return token_count, file_size
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Count tokens in DCLM dataset")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Base directory where the data is stored"
    )
    args = parser.parse_args()

    # Get the data mix configuration
    tokenizer_config = TokenizerConfig.dolma2()
    data_mix = DataMix.OLMo_dclm

    # Build the list of files
    paths, labels = data_mix.build(args.data_root, str(tokenizer_config.identifier))

    print(f"Data mix: {data_mix}")
    print(f"Tokenizer: {tokenizer_config.identifier}")
    print(f"Number of files: {len(paths)}")
    print(f"\nCounting tokens in each file...")

    total_tokens = 0
    total_size = 0
    successful_files = 0
    remote_files = 0

    for i, (path, label) in enumerate(zip(paths, labels), 1):
        print(f"\n[{i}/{len(paths)}] {label}")
        token_count, file_size = count_tokens_in_file(path)

        if token_count is not None:
            total_tokens += token_count
            successful_files += 1
        if file_size is not None:
            total_size += file_size
            if is_url(path):
                remote_files += 1

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files: {len(paths)}")
    print(f"Successfully counted: {successful_files}")
    print(f"Remote files: {remote_files}")

    if successful_files > 0:
        print(f"\nTotal tokens: {total_tokens:,}")
        print(f"Total tokens (scientific): {total_tokens:.2e}")
        print(f"Total size: {total_size:,} bytes ({total_size / 1024**3:.2f} GB)")

    if remote_files > 0 and successful_files == 0:
        print("\nAll files are remote. To get exact token counts, you need to:")
        print("1. Download the files locally, or")
        print("2. Use the file sizes to estimate (assuming uint16 dtype, divide bytes by 2)")
        if total_size > 0:
            estimated_tokens = total_size // 2  # Assuming uint16
            print(f"\nEstimated tokens (assuming uint16): {estimated_tokens:,} ({estimated_tokens:.2e})")

if __name__ == "__main__":
    main()
