"""
Proper evaluation script that uses the training framework to load checkpoints and evaluate.
Based on how the training script does evaluation during training.
"""

import argparse
import json
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
from olmo_core.train import Duration, prepare_training_environment, teardown_training_environment
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig
from olmo_core.utils import get_default_device, move_to_device, seed_all
from olmo_core.distributed.utils import is_distributed


def evaluate_checkpoint(checkpoint_path: str, work_dir: str, sequence_length: int = 4096) -> dict:
    """Evaluate a single checkpoint on the validation set using the proper training framework."""

    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        # Setup configs (same as training script)
        tokenizer_config = TokenizerConfig.dolma2()
        model_config = TransformerConfig.olmo2_30M(
            vocab_size=tokenizer_config.padded_vocab_size(),
        )

        # Build the model
        model = model_config.build(init_device="cpu")

        # Create train module config (minimal, just for loading)
        train_module_config = TransformerTrainModuleConfig(
            rank_microbatch_size=1 * 4096,  # Reduced to avoid OOM
            max_sequence_length=sequence_length,
            optim=AdamWConfig(lr=1e-3),  # Dummy optim, won't be used
        )

        # Build train module with the model
        train_module = train_module_config.build(model)

        # Load checkpoint
        checkpointer = Checkpointer(work_dir=Path(work_dir))
        checkpointer.load(
            checkpoint_path,
            train_module,
            load_trainer_state=False,
            load_optim_state=False,
        )

        # Setup validation dataset
        eval_dataset_config = NumpyPaddedFSLDatasetConfig.from_data_mix(
            DataMix.dclm_validation,
            mix_base_dir="/n/netscratch/dam_lab/Lab/sqin/olmo",
            sequence_length=sequence_length,
            tokenizer=tokenizer_config,
            work_dir=work_dir,
        )

        # Build dataset
        eval_dataset = eval_dataset_config.build()

        # Create val data loader
        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=4 * 4096,  # Reduced to avoid OOM on 40GB GPU
            seed=42,
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues
        )
        data_loader = data_loader_config.build(
            eval_dataset,
            dp_process_group=train_module.dp_process_group if is_distributed() else None
        )

        # Reshuffle the data loader (required before iterating)
        data_loader.reshuffle(epoch=1)

        # Move to device and set eval mode
        device = get_default_device()
        train_module.model.to("cuda")
        train_module.model.eval()

        # Ensure gradients are disabled for all parameters
        for param in train_module.model.parameters():
            param.requires_grad = False

        # Run evaluation
        print("Evaluating...")
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        max_tokens = 50_000_000  # Evaluate on 50M tokens

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for batch in data_loader:
                if total_tokens >= max_tokens:
                    break

                batch = move_to_device(batch, device)
                labels = get_labels(batch)

                # Use train_module.eval_batch() like the training callback does
                output = train_module.eval_batch(batch, labels=labels)

                # output is LMOutputWithLoss: (logits, loss, ce_loss, z_loss)
                _, loss, ce_loss, _ = output

                # Count valid tokens
                valid_mask = (labels != -100)
                batch_tokens = valid_mask.sum().item()

                total_loss += ce_loss.sum().item()  # ce_loss is unreduced
                total_tokens += batch_tokens
                num_batches += 1

                if num_batches % 200 == 0:
                    avg_loss = total_loss / total_tokens
                    print(f"  Batch {num_batches}: Loss={avg_loss:.4f}, Tokens={total_tokens:,}")

        # Final metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"✓ Final: Loss={avg_loss:.4f}, PPL={perplexity:.2f}, Tokens={total_tokens:,}")

        result = {
            "validation_loss": avg_loss,
            "perplexity": perplexity,
            "tokens_evaluated": total_tokens,
            "num_batches": num_batches,
        }

        return result

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Full path to checkpoint to evaluate (e.g., /path/to/30M_run1/step4578)")
    parser.add_argument("--run_name", type=str, required=True,
                       help="Name of the run (for results tracking)")
    parser.add_argument("--work_dir", type=str, default="/n/netscratch/dam_lab/Lab/sqin/olmo/dataset-cache",
                       help="Working directory for dataset cache")
    parser.add_argument("--sequence_length", type=int, default=4096,
                       help="Sequence length (must match training)")
    parser.add_argument("--output", type=str, default="eval_results.json",
                       help="Output JSON file")

    args = parser.parse_args()

    # Prepare environment (but don't initialize distributed)
    seed_all(42)

    checkpoint_path = Path(args.checkpoint_path)
    run_name = args.run_name
    output_path = Path(args.output)

    # Load existing results if output file exists
    results = {}
    if output_path.exists():
        print(f"Loading existing results from: {output_path}")
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Found {len(results)} existing results")

    # Skip if already evaluated (and not an error)
    if run_name in results and "error" not in results[run_name]:
        print(f"✓ Skipping {run_name}: already evaluated")
        print(f"  Loss={results[run_name]['validation_loss']:.4f}, PPL={results[run_name]['perplexity']:.2f}")
        return

    if not checkpoint_path.exists():
        print(f"⚠ Error: Checkpoint not found at {checkpoint_path}")
        results[run_name] = {"error": f"Checkpoint not found at {checkpoint_path}"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return

    print(f"\n{'='*80}")
    print(f"Evaluating: {run_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}")

    try:
        result = evaluate_checkpoint(
            str(checkpoint_path),
            args.work_dir,
            args.sequence_length
        )
        results[run_name] = result

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {output_path}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results[run_name] = {"error": str(e)}

        # Save error result too
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
