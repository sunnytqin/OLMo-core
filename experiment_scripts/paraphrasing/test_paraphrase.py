#!/usr/bin/env python3
"""
Test paraphrasing tokenized .npy data using Qwen/Qwen3-4B-Instruct-2507.

Loads tokenized data, decodes documents to text, paraphrases each with Qwen3,
and prints original vs paraphrased side by side.
"""

import argparse
import os

import numpy as np
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/n/netscratch/barak_lab/Lab/sqin/cache"

EOS_TOKEN_ID = 100257

SYSTEM_PROMPT = "Provide direct and detailed response to the instructions without adding additional notes."

USER_PROMPT_TEMPLATE = (
    "For the following document, regardless of its original content or formatting, "
    "write a full article of the same content in high quality English language as in "
    "texts on Wikipedia: {text}. Provide the rephrased article without any additional "
    "notes. Long article with full length and complete details. Rephrased article:"
)

def load_dolma2_tokenizer():
    """Load the dolma2 tokenizer for decoding the .npy data."""
    return Tokenizer.from_pretrained("allenai/dolma2-tokenizer")


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
        doc_tokens = arr[doc_start:eos_pos].tolist()
        documents.append(doc_tokens)
        doc_start = eos_pos + 1

    return documents


def load_paraphrase_model(device="auto"):
    """Load Qwen/Qwen3-4B-Instruct-2507 for paraphrasing."""
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
    )
    return model, tokenizer


def paraphrase(text, model, tokenizer, max_input_chars=2000, max_new_tokens=2048):
    """Paraphrase a document using Qwen3-4B-Instruct."""
    # Truncate very long documents for this test
    truncated = text[:max_input_chars]
    if len(text) > max_input_chars:
        truncated += "..."

    user_msg = USER_PROMPT_TEMPLATE.format(text=truncated)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    # Decode only the generated part (skip the prompt tokens)
    output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
    result = tokenizer.decode(output_ids, skip_special_tokens=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Test paraphrasing on tokenized data")
    parser.add_argument(
        "--input",
        type=str,
        default="/n/netscratch/dam_lab/Lab/sqin/olmo/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/resharded/train_0.3B.npy",
        help="Path to the tokenized .npy file",
    )
    parser.add_argument("--num-docs", type=int, default=3, help="Number of documents to paraphrase")
    parser.add_argument("--max-input-chars", type=int, default=2000, help="Max chars of original text to feed to the model")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="Device for the model")
    args = parser.parse_args()

    # Load dolma2 tokenizer for decoding the .npy data
    print("Loading dolma2 tokenizer...")
    dolma2_tok = load_dolma2_tokenizer()

    # Load the .npy data
    print(f"Loading data from: {args.input}")
    arr = np.memmap(args.input, dtype=np.uint32, mode="r")

    print("Extracting documents...")
    documents = extract_documents(arr, max_docs=args.num_docs)

    # Load Qwen3 model for paraphrasing
    model, model_tok = load_paraphrase_model(device=args.device)
    print()

    # Paraphrase each document
    for i, doc_tokens in enumerate(documents):
        original_text = dolma2_tok.decode(doc_tokens, skip_special_tokens=False)

        print("=" * 80)
        print(f"DOCUMENT {i+1} ({len(doc_tokens):,} tokens, {len(original_text):,} chars)")
        print("=" * 80)

        print(f"\n--- ORIGINAL ---")
        preview = original_text[:args.max_input_chars]
        if len(original_text) > args.max_input_chars:
            preview += "..."
        print(preview)

        print(f"\n--- PARAPHRASED ---")
        paraphrased = paraphrase(
            original_text,
            model,
            model_tok,
            max_input_chars=args.max_input_chars,
            max_new_tokens=args.max_new_tokens,
        )
        print(paraphrased)
        print()


if __name__ == "__main__":
    main()
