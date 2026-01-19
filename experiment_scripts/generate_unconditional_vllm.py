#!/usr/bin/env python3
"""
Unconditional text generation using vLLM for synthetic training data.

Generates sequences starting from EOS token (to match training distribution)
and saves output in the same format as training data (.npy files with uint32).
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using vLLM"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/370M_seed42/step883_hf",
        help="Path to the HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_data",
        help="Directory to save generated sequences",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=7_000_000_000,
        help="Total number of tokens to generate (default: 7B)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
        help="Length of each generated sequence in tokens",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=192,
        help="Number of sequences to generate per batch",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--tokens-per-file",
        type=int,
        default=100_000_000,
        help="Approximate number of tokens per output file (default: 100M)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard ID for parallel generation (0-indexed)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for parallel generation",
    )
    return parser.parse_args()


class GenerationCheckpoint:
    """Manages checkpointing for resumable generation."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.sequences_generated = 0
        self.tokens_generated = 0
        self.current_file_idx = 0
        self.tokens_in_current_file = 0

    def load(self) -> bool:
        """Load checkpoint if exists. Returns True if loaded successfully."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
            self.sequences_generated = data['sequences_generated']
            self.tokens_generated = data['tokens_generated']
            self.current_file_idx = data['current_file_idx']
            self.tokens_in_current_file = data['tokens_in_current_file']
            return True
        return False

    def save(self):
        """Save current checkpoint."""
        data = {
            'sequences_generated': self.sequences_generated,
            'tokens_generated': self.tokens_generated,
            'current_file_idx': self.current_file_idx,
            'tokens_in_current_file': self.tokens_in_current_file,
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)


def save_sequences_to_numpy(
    sequences: List[np.ndarray],
    output_path: Path,
    eos_token_id: int,
    append: bool = False,
):
    """
    Save sequences to a numpy file in training data format.

    Format: Concatenated token sequences as raw uint32 binary (matching training data).
    Sequences are exactly sequence_length tokens (e.g., 4096) to align with training.
    """
    # Concatenate all sequences
    concatenated = np.concatenate(sequences)

    # Save as uint32 to match training data format
    # Note: We save as raw binary (tofile), not .npy format
    concatenated_u32 = concatenated.astype(np.uint32)

    if append and output_path.exists():
        # Append to existing file
        with open(output_path, 'ab') as f:
            concatenated_u32.tofile(f)
    else:
        # Write new file
        concatenated_u32.tofile(output_path)


def main():
    args = parse_args()

    # Validate sharding parameters
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id must be in range [0, {args.num_shards})")

    # Calculate tokens per shard
    tokens_per_shard = args.total_tokens // args.num_shards
    shard_seed = args.seed + args.shard_id  # Different seed per shard for diversity

    # Setup output directory (shard-specific subdirectory)
    if args.num_shards > 1:
        output_dir = Path(args.output_dir) / f"shard_{args.shard_id}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup checkpoint (shard-specific)
    checkpoint = GenerationCheckpoint(output_dir / "checkpoint.json")
    if args.resume and checkpoint.load():
        print(f"Resuming from checkpoint:")
        print(f"  Sequences generated: {checkpoint.sequences_generated:,}")
        print(f"  Tokens generated: {checkpoint.tokens_generated:,}")
        print(f"  Current file index: {checkpoint.current_file_idx}")
    else:
        print("Starting fresh generation")

    # Calculate generation requirements (for this shard)
    total_sequences = (tokens_per_shard + args.sequence_length - 1) // args.sequence_length
    remaining_sequences = total_sequences - checkpoint.sequences_generated

    print(f"\nGeneration Configuration:")
    print(f"  Model: {args.model_path}")
    if args.num_shards > 1:
        print(f"  Shard: {args.shard_id} / {args.num_shards}")
        print(f"  Total tokens (all shards): {args.total_tokens:,}")
        print(f"  Tokens for this shard: {tokens_per_shard:,}")
    else:
        print(f"  Total tokens to generate: {args.total_tokens:,}")
    print(f"  Sequence length: {args.sequence_length:,}")
    print(f"  Total sequences needed: {total_sequences:,}")
    print(f"  Remaining sequences: {remaining_sequences:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Tokens per file: {args.tokens_per_file:,}")
    print(f"  Seed: {shard_seed}")

    if remaining_sequences <= 0:
        print("\nGeneration already complete!")
        return

    # Initialize vLLM
    print(f"\nInitializing vLLM...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        seed=shard_seed,
        max_model_len=args.sequence_length,  # +1 for initial EOS
    )

    # Get tokenizer and special tokens
    tokenizer = llm.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id
    print(f"  EOS token ID: {eos_token_id}")

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.sequence_length,
        ignore_eos=True,  # Keep generating even when EOS is produced
        # Note: seed is set at LLM level, not per-request, to ensure diverse outputs
    )

    # Prepare initial prompt (single EOS token)
    # We'll generate from EOS but strip it from the output
    prompt_text = tokenizer.decode([eos_token_id])

    # Generation loop
    print(f"\nStarting generation...")
    start_time = time.time()
    sequences_generated = 0

    # Buffer for accumulating sequences before writing to file
    sequence_buffer = []

    with tqdm(total=remaining_sequences, desc="Generating sequences") as pbar:
        while sequences_generated < remaining_sequences:
            # Determine batch size (might be smaller for last batch)
            current_batch_size = min(args.batch_size, remaining_sequences - sequences_generated)

            # Prepare batch of prompts (all identical)
            prompts = [prompt_text] * current_batch_size

            # Generate batch
            batch_start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            batch_time = time.time() - batch_start

            # Process outputs
            for output in outputs:
                # Get generated token IDs
                generated_ids = output.outputs[0].token_ids

                # Convert to numpy array (uint32 to match training data format)
                token_array = np.array(generated_ids, dtype=np.uint32)

                sequence_buffer.append(token_array)
                checkpoint.tokens_in_current_file += len(token_array)
                checkpoint.tokens_generated += len(token_array)

            sequences_generated += current_batch_size
            checkpoint.sequences_generated += current_batch_size

            # Update progress bar with stats
            elapsed = time.time() - start_time
            tokens_per_sec = checkpoint.tokens_generated / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                'tokens/s': f'{tokens_per_sec:.1f}',
                'batch_time': f'{batch_time:.2f}s',
                'file': checkpoint.current_file_idx,
            })
            pbar.update(current_batch_size)

            # Save to file if buffer is large enough
            if checkpoint.tokens_in_current_file >= args.tokens_per_file:
                output_path = output_dir / f"output_{checkpoint.current_file_idx:06d}.npy"
                print(f"\n  Saving to {output_path.name} ({len(sequence_buffer)} sequences, {checkpoint.tokens_in_current_file:,} tokens)")
                save_sequences_to_numpy(sequence_buffer, output_path, eos_token_id)

                # Reset for next file
                sequence_buffer = []
                checkpoint.current_file_idx += 1
                checkpoint.tokens_in_current_file = 0

                # Save checkpoint
                checkpoint.save()

    # Save any remaining sequences
    if sequence_buffer:
        output_path = output_dir / f"output_{checkpoint.current_file_idx:06d}.npy"
        print(f"\n  Saving final file {output_path.name} ({len(sequence_buffer)} sequences, {checkpoint.tokens_in_current_file:,} tokens)")
        save_sequences_to_numpy(sequence_buffer, output_path, eos_token_id)
        checkpoint.save()

    # Final statistics
    total_time = time.time() - start_time
    total_tokens_per_sec = checkpoint.tokens_generated / total_time

    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"  Total sequences: {checkpoint.sequences_generated:,}")
    print(f"  Total tokens: {checkpoint.tokens_generated:,}")
    print(f"  Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    print(f"  Average throughput: {total_tokens_per_sec:.1f} tokens/sec")
    print(f"  Output files: {checkpoint.current_file_idx + 1}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
