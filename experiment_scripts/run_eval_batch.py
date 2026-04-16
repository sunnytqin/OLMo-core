#!/usr/bin/env python3
"""
Batch evaluation script that loads model architecture + data loader once,
then swaps checkpoint weights for each run. Much faster than spawning
a subprocess per checkpoint.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.utils import get_labels
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig
from olmo_core.utils import get_default_device, move_to_device, seed_all

_MODEL_CONFIGS = {
    "30M": TransformerConfig.olmo3_30M,
    "60M": TransformerConfig.olmo3_60M,
    "190M": TransformerConfig.olmo3_190M,
    "370M": TransformerConfig.olmo3_370M,
}


def setup_eval(model_size: str, sequence_length: int, work_dir: str,
               max_tokens: int = 13_000_000):
    """Build model, train_module, and cache eval batches once."""
    tokenizer_config = TokenizerConfig.dolma2()
    config_fn = _MODEL_CONFIGS[model_size]
    model_config = config_fn(vocab_size=tokenizer_config.padded_vocab_size())
    model = model_config.build(init_device="cpu")

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=1 * sequence_length,
        max_sequence_length=sequence_length,
        optim=AdamWConfig(lr=1e-3),
    )
    train_module = train_module_config.build(model)

    # Data loader
    gbs = 4 * sequence_length
    eval_dataset_config = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_dolma_val,
        mix_base_dir="/n/netscratch/barak_lab/Everyone/sqin/olmo",
        sequence_length=sequence_length,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
    )
    eval_dataset = eval_dataset_config.build()
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=gbs,
        seed=42,
        num_workers=0,
    )
    data_loader = data_loader_config.build(eval_dataset, dp_process_group=None)

    # Cache eval batches: load once, reuse for every checkpoint
    print(f"Caching eval batches (up to {max_tokens:,} data tokens)...")
    data_loader.reshuffle(epoch=1)
    eval_batches = []
    data_tokens = 0
    for batch in data_loader:
        eval_batches.append(batch)
        data_tokens += batch["input_ids"].numel()
        if data_tokens >= max_tokens:
            break
    print(f"Cached {len(eval_batches)} batches ({data_tokens:,} data tokens)")

    checkpointer = Checkpointer(work_dir=Path(work_dir))
    device = get_default_device()

    return train_module, eval_batches, checkpointer, device


def evaluate_one(train_module, eval_batches, checkpointer, device,
                 checkpoint_path: str) -> dict:
    """Load weights from checkpoint_path and evaluate on cached batches."""
    train_module.model.to("cpu")
    checkpointer.load(
        checkpoint_path,
        train_module,
        load_trainer_state=False,
        load_optim_state=False,
    )
    train_module.model.to(device)
    train_module.model.eval()

    total_loss = 0.0
    total_valid_tokens = 0

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in eval_batches:
            batch = move_to_device(batch, device)
            labels = get_labels(batch)
            output = train_module.eval_batch(batch, labels=labels)
            _, _, ce_loss, _ = output
            total_loss += ce_loss.sum().item()
            total_valid_tokens += (labels != -100).sum().item()

    avg_loss = total_loss / total_valid_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "validation_loss": avg_loss,
        "perplexity": perplexity,
        "tokens_evaluated": total_valid_tokens,
        "num_batches": len(eval_batches),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="../results/hparam/manifest.json")
    parser.add_argument("--results-dir", type=str, default="../results/dolma_val_loss")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--work-dir", type=str,
                        default="/n/netscratch/dam_lab/Lab/sqin/olmo/dataset-cache")
    parser.add_argument("--sequence-length", type=int, default=4096)
    args = parser.parse_args()

    seed_all(42)

    # Load manifest and get this worker's slice
    with open(args.manifest) as f:
        manifest = json.load(f)
    complete = [c for c in manifest["checkpoints"] if c.get("complete", True)]
    total = len(complete)
    chunk = (total + args.num_workers - 1) // args.num_workers
    start = args.worker_id * chunk
    end = min(start + chunk, total)
    entries = complete[start:end]
    print(f"Worker {args.worker_id}: processing {len(entries)} checkpoints "
          f"(indices {start}-{end-1} of {total})")

    # Group entries by model_size so we only rebuild once per size
    from collections import defaultdict
    by_size = defaultdict(list)
    for entry in entries:
        by_size[entry["model_size"]].append(entry)

    results_dir = Path(args.results_dir)
    processed = 0
    skipped = 0
    errors = 0

    for model_size, size_entries in sorted(by_size.items()):
        print(f"\n{'='*60}")
        print(f"Setting up {model_size} model ({len(size_entries)} checkpoints)")
        print(f"{'='*60}")

        train_module, eval_batches, checkpointer, device = setup_eval(
            model_size, args.sequence_length, args.work_dir
        )

        for i, entry in enumerate(size_entries):
            run_name = entry["run_name"]
            chin_dir = entry["chinchilla_dir"]
            checkpoint_path = entry["checkpoint_path"]

            out_dir = results_dir / chin_dir / model_size
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{run_name}.json"

            # Skip if already done
            if out_file.exists():
                try:
                    with open(out_file) as f:
                        existing = json.load(f)
                    if run_name in existing and "error" not in existing[run_name]:
                        skipped += 1
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass

            t0 = time.time()
            print(f"[{processed+skipped+1}/{len(size_entries)}] {run_name} ...", end=" ", flush=True)

            try:
                result = evaluate_one(
                    train_module, eval_batches, checkpointer, device,
                    checkpoint_path,
                )
                elapsed = time.time() - t0
                print(f"loss={result['validation_loss']:.4f} ({elapsed:.0f}s)")

                with open(out_file, "w") as f:
                    json.dump({run_name: result}, f, indent=2)
                processed += 1

            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR ({elapsed:.0f}s): {e}")
                with open(out_file, "w") as f:
                    json.dump({run_name: {"error": str(e)}}, f, indent=2)
                errors += 1

            torch.cuda.empty_cache()

        # Free GPU memory before loading next model size
        del train_module, eval_batches, checkpointer
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Worker {args.worker_id} done: {processed} evaluated, "
          f"{skipped} skipped, {errors} errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
