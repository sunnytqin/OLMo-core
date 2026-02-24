#!/usr/bin/env python3
"""
Paraphrase a shard of documents from tokenized .npy data using vLLM.

Each SLURM array job runs this script with a different --shard-id.
The script:
1. Loads the tokenized .npy and extracts its shard of documents
2. Decodes tokens to text (dolma2 tokenizer)
3. Paraphrases using vLLM (OLMo-3-7B-Instruct) in batches
4. Re-tokenizes paraphrased text (dolma2 tokenizer)
5. Saves as a shard .npy file
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer as DolmaTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["HF_HOME"] = "/n/netscratch/barak_lab/Lab/sqin/cache"

EOS_TOKEN_ID = 100257
MODEL_NAME = "allenai/OLMo-3-7B-Instruct"

SYSTEM_PROMPT = "Provide direct and detailed response to the instructions without adding additional notes."

USER_PROMPT_TEMPLATE = (
    "For the following document, regardless of its original content or formatting, "
    "write a full article of the same content in high quality English language as in "
    "texts on Wikipedia: {text}. Provide the rephrased article without any additional "
    "notes. Long article with full length and complete details. Rephrased article:"
)


def load_dolma2_tokenizer():
    """Load the dolma2 tokenizer for decoding/re-encoding .npy data."""
    return DolmaTokenizer.from_pretrained("allenai/dolma2-tokenizer")


def extract_shard_documents(arr, shard_id, num_shards, subsample=1):
    """Extract documents for this shard from the flat token array.

    Args:
        subsample: Only take every Nth document (e.g., 2 = every other doc).
            This is applied globally before sharding so the same documents
            are selected regardless of num_shards.
    """
    eos_positions = np.where(arr == EOS_TOKEN_ID)[0]
    total_docs = len(eos_positions)

    # Subsample: take every Nth document (globally, before sharding)
    selected_indices = list(range(0, total_docs, subsample))
    num_selected = len(selected_indices)

    # Shard the selected documents
    docs_per_shard = num_selected // num_shards
    shard_start = shard_id * docs_per_shard
    shard_end = shard_start + docs_per_shard if shard_id < num_shards - 1 else num_selected

    print(f"Total documents: {total_docs:,}")
    print(f"Subsample every {subsample} → {num_selected:,} docs selected")
    print(f"Shard {shard_id}/{num_shards}: selected docs [{shard_start:,}, {shard_end:,}) = {shard_end - shard_start:,} docs")

    documents = []
    for s in range(shard_start, shard_end):
        i = selected_indices[s]  # global doc index
        doc_start = 0 if i == 0 else int(eos_positions[i - 1]) + 1
        doc_end = int(eos_positions[i])
        doc_tokens = arr[doc_start:doc_end].tolist()
        documents.append(doc_tokens)

    return documents


def format_prompt(text, model_tokenizer):
    """Format a single chat prompt for paraphrasing."""
    user_msg = USER_PROMPT_TEMPLATE.format(text=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return model_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def tokenize_to_npy(texts, dolma2_tok, output_path):
    """Re-tokenize paraphrased texts and save as .npy (uint32 memmap)."""
    all_tokens = []
    for text in texts:
        if not text or not text.strip():
            continue
        token_ids = dolma2_tok.encode(text.strip()).ids
        token_ids.append(EOS_TOKEN_ID)
        all_tokens.extend(token_ids)

    arr = np.array(all_tokens, dtype=np.uint32)
    memmap = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=arr.shape)
    memmap[:] = arr[:]
    memmap.flush()
    del memmap
    print(f"Saved {len(arr):,} tokens to {output_path}")
    return len(arr)


def load_checkpoint(checkpoint_path):
    """Load paraphrased texts from checkpoint JSONL."""
    results = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                entry = json.loads(line)
                results[entry["idx"]] = entry["text"]
    return results


def save_checkpoint_batch(checkpoint_path, batch_results):
    """Append a batch of results to the checkpoint JSONL."""
    with open(checkpoint_path, "a") as f:
        for idx, text in batch_results:
            f.write(json.dumps({"idx": idx, "text": text}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Paraphrase a shard of documents using vLLM")
    parser.add_argument(
        "--input", type=str,
        default="/n/netscratch/dam_lab/Lab/sqin/olmo/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/resharded/train_0.3B.npy",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for shard .npy files")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--subsample", type=int, default=2, help="Paraphrase every Nth document (2 = every other doc for ~2:1 ratio)")
    parser.add_argument("--max-model-len", type=int, default=16384, help="Max total sequence length (input + output) for vLLM")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of documents to process per vLLM batch")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"shard_{args.shard_id:04d}.jsonl"
    output_npy_path = output_dir / f"shard_{args.shard_id:04d}.npy"

    # --- 1. Load data and extract shard ---
    print(f"Loading dolma2 tokenizer...")
    dolma2_tok = load_dolma2_tokenizer()

    print(f"Loading data: {args.input}")
    arr = np.memmap(args.input, dtype=np.uint32, mode="r")

    print(f"Extracting shard documents...")
    documents = extract_shard_documents(arr, args.shard_id, args.num_shards, subsample=args.subsample)

    # Decode all documents to text
    print(f"Decoding {len(documents):,} documents to text...")
    doc_texts = []
    for doc_tokens in tqdm(documents, desc="Decoding"):
        doc_texts.append(dolma2_tok.decode(doc_tokens, skip_special_tokens=False))
    del documents  # free memory

    # --- 2. Check checkpoint ---
    completed = {}
    if args.resume:
        completed = load_checkpoint(checkpoint_path)
        print(f"Checkpoint: {len(completed):,} documents already paraphrased")

    remaining_indices = [i for i in range(len(doc_texts)) if i not in completed]
    print(f"Remaining: {len(remaining_indices):,} documents to paraphrase")

    if not remaining_indices:
        print("All documents already paraphrased. Tokenizing and saving...")
        paraphrased_texts = [completed[i] for i in range(len(doc_texts))]
        tokenize_to_npy(paraphrased_texts, dolma2_tok, output_npy_path)
        return

    # --- 3. Initialize vLLM ---
    print(f"\nLoading model: {MODEL_NAME}")
    model_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=args.max_model_len,
    )

    # --- 4. Process in chunks ---
    start_time = time.time()
    total_processed = 0
    skipped = 0

    for batch_start in range(0, len(remaining_indices), args.batch_size):
        batch_indices = remaining_indices[batch_start : batch_start + args.batch_size]
        batch_texts = [doc_texts[i] for i in batch_indices]

        # Format prompts, skip docs whose prompt exceeds max_model_len
        prompts = []
        valid_indices = []
        for idx, text in zip(batch_indices, batch_texts):
            prompt = format_prompt(text, model_tokenizer)
            prompt_tokens = len(model_tokenizer.encode(prompt))
            if prompt_tokens >= args.max_model_len:
                print(f"  Skipping doc {idx} (prompt={prompt_tokens:,} tokens >= max_model_len)")
                skipped += 1
                continue
            prompts.append(prompt)
            valid_indices.append(idx)

        if not prompts:
            continue

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Collect results
        batch_results = []
        for idx, output in zip(valid_indices, outputs):
            paraphrased = output.outputs[0].text
            batch_results.append((idx, paraphrased))

        # Save checkpoint
        save_checkpoint_batch(checkpoint_path, batch_results)

        total_processed += len(batch_indices)
        elapsed = time.time() - start_time
        docs_per_sec = total_processed / elapsed
        remaining = len(remaining_indices) - total_processed
        eta = remaining / docs_per_sec if docs_per_sec > 0 else 0

        print(
            f"  Batch done: {total_processed:,}/{len(remaining_indices):,} docs | "
            f"{docs_per_sec:.1f} docs/s | ETA: {eta/60:.1f} min"
        )

    # --- 5. Tokenize and save final .npy ---
    print(f"\nLoading all paraphrased texts from checkpoint...")
    completed = load_checkpoint(checkpoint_path)
    paraphrased_texts = [completed[i] for i in range(len(doc_texts)) if i in completed]

    print(f"  {len(paraphrased_texts):,} paraphrased, {len(doc_texts) - len(paraphrased_texts):,} skipped (too long)")
    print(f"Re-tokenizing and saving to {output_npy_path}...")
    total_tokens = tokenize_to_npy(paraphrased_texts, dolma2_tok, output_npy_path)

    total_time = time.time() - start_time
    print(f"\nDone! Shard {args.shard_id}")
    print(f"  Documents paraphrased: {len(paraphrased_texts):,} / {len(doc_texts):,}")
    print(f"  Output tokens: {total_tokens:,}")
    print(f"  Time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
