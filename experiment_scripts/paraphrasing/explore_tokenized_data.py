#!/usr/bin/env python3
"""
Explore tokenized .npy data: load, decode back to text, and inspect documents.

The .npy files are flat 1D arrays of uint32 token IDs. Documents are concatenated
with EOS tokens (ID=100257) between them.
"""

import argparse
import numpy as np
from tokenizers import Tokenizer


EOS_TOKEN_ID = 100257


def load_tokenizer():
    """Load the dolma2 tokenizer from HuggingFace."""
    tokenizer = Tokenizer.from_pretrained("allenai/dolma2-tokenizer")
    return tokenizer


def extract_documents(arr, max_docs=None):
    """Extract individual documents from the flat token array, split by EOS tokens."""
    eos_positions = np.where(arr == EOS_TOKEN_ID)[0]
    print(f"Total tokens: {len(arr):,}")
    print(f"Total EOS tokens (documents): {len(eos_positions):,}")
    print()

    documents = []
    doc_start = 0
    for i, eos_pos in enumerate(eos_positions):
        if max_docs is not None and i >= max_docs:
            break
        doc_tokens = arr[doc_start:eos_pos].tolist()  # exclude EOS itself
        documents.append(doc_tokens)
        doc_start = eos_pos + 1  # skip past EOS

    return documents


def main():
    parser = argparse.ArgumentParser(description="Explore tokenized .npy data")
    parser.add_argument(
        "--input",
        type=str,
        default="/n/netscratch/dam_lab/Lab/sqin/olmo/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/resharded/train_0.3B.npy",
        help="Path to the tokenized .npy file",
    )
    parser.add_argument("--num-docs", type=int, default=5, help="Number of documents to display")
    parser.add_argument("--max-chars", type=int, default=500, help="Max characters to display per document (0 for full)")
    args = parser.parse_args()

    print(f"Loading tokenizer...")
    tokenizer = load_tokenizer()

    print(f"Loading data from: {args.input}")
    arr = np.memmap(args.input, dtype=np.uint32, mode="r")

    print(f"Extracting documents...")
    documents = extract_documents(arr, max_docs=args.num_docs)

    print(f"Decoded {len(documents)} documents:")
    print("=" * 80)
    for i, doc_tokens in enumerate(documents):
        text = tokenizer.decode(doc_tokens, skip_special_tokens=False)
        print(f"\n--- Document {i+1} ({len(doc_tokens):,} tokens, {len(text):,} chars) ---")
        if args.max_chars > 0 and len(text) > args.max_chars:
            print(text[: args.max_chars] + "...")
        else:
            print(text)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
