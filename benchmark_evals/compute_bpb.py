"""
Compute bytes-per-token ratio for the dolma2 tokenizer on a text sample,
convert cross-entropy validation losses (nats) to bits-per-byte (BPB),
and select best runs from hyperparameter sweeps.

Usage:
    # Compute bytes_per_token ratio on a dataset sample
    python compute_bpb.py --measure-ratio

    # Convert a CE loss value to BPB
    python compute_bpb.py --ce-loss 4.12

    # Show best runs for 370M models
    python compute_bpb.py --best-runs --size 370M

    # Export best runs as JSON
    python compute_bpb.py --best-runs --size 370M --export best_runs_370m.json
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# Measured on C4 validation data with allenai/dolma2-tokenizer (5000 samples).
# Re-measure with --measure-ratio if your validation data distribution differs.
DEFAULT_BYTES_PER_TOKEN = 4.6954

REPO_ROOT = Path(__file__).parent.parent
DOLMA_VAL_LOSS_DIR = REPO_ROOT / "results" / "dolma_val_loss"

RUN_PATTERN = re.compile(
    r'(\d+M)_seed\d+_case4_dolma_epoch(\d+)_wd([\d.]+)_lr([\d.e-]+)'
)


def measure_bytes_per_token(num_samples: int = 5000) -> float:
    """Measure bytes-per-token ratio by tokenizing a sample of C4 validation data."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    total_bytes = 0
    total_tokens = 0

    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        text = example["text"]
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text))

    ratio = total_bytes / total_tokens
    print(f"Measured bytes_per_token = {ratio:.4f}")
    print(f"  (from {num_samples} samples, {total_tokens} tokens, {total_bytes} bytes)")
    return ratio


def ce_to_bpb(ce_loss_nats: float, bytes_per_token: float = DEFAULT_BYTES_PER_TOKEN) -> float:
    """Convert cross-entropy loss (nats/token) to bits-per-byte.

    BPB = CE_loss_nats / ln(2) / bytes_per_token
        = bits_per_token / bytes_per_token
    """
    bits_per_token = ce_loss_nats / math.log(2)
    return bits_per_token / bytes_per_token


def select_best_runs(val_loss_dir: Path = DOLMA_VAL_LOSS_DIR,
                     size_filter: str = None) -> list[dict]:
    """Select the best (lr, wd) for each (chinchilla_scale, model_size, epoch).

    Reads individual JSON files from results/dolma_val_loss/{chinchilla}/{size}/.
    Returns a list of dicts sorted by (size, chinchilla_scale, epoch).
    """
    # Collect all results: (chin, size, epoch) -> {(wd, lr): val_loss}
    grid = defaultdict(dict)

    for chin_dir in sorted(val_loss_dir.iterdir()):
        if not chin_dir.is_dir() or not chin_dir.name.startswith("chinchilla_"):
            continue
        chin_scale = float(chin_dir.name.replace("chinchilla_", ""))

        for size_dir in sorted(chin_dir.iterdir()):
            if not size_dir.is_dir():
                continue
            size = size_dir.name
            if size_filter and size != size_filter:
                continue

            for json_file in sorted(size_dir.glob("*.json")):
                m = RUN_PATTERN.search(json_file.stem)
                if not m:
                    continue
                _, epoch, wd, lr = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))

                with open(json_file) as f:
                    data = json.load(f)
                # JSON has one top-level key with metrics inside
                metrics = next(iter(data.values()))
                if "error" in metrics:
                    continue
                val_loss = metrics.get("validation_loss")
                if val_loss is not None:
                    grid[(chin_scale, size, epoch)][(wd, lr)] = val_loss

    # Pick best (wd, lr) per (chin, size, epoch)
    records = []
    for (chin, size, epoch), hparams in sorted(grid.items()):
        best_key = min(hparams, key=hparams.get)
        best_wd, best_lr = best_key
        records.append({
            "chinchilla_scale": chin,
            "size": size,
            "epoch": epoch,
            "flops_multiplier": chin * epoch,
            "validation_loss": hparams[best_key],
            "bpb": ce_to_bpb(hparams[best_key]),
            "learning_rate": best_lr,
            "weight_decay": best_wd,
        })

    return records


def print_best_runs(records: list[dict]):
    """Print best runs as a formatted table."""
    print(f"{'Size':>5} {'Chin':>6} {'Epoch':>5} {'FLOPs':>7} "
          f"{'Val Loss':>9} {'BPB':>7} {'LR':>9} {'WD':>5}")
    print("-" * 65)
    for r in records:
        print(f"{r['size']:>5} {r['chinchilla_scale']:>6.2f} {r['epoch']:>5d} "
              f"{r['flops_multiplier']:>7.2f} {r['validation_loss']:>9.4f} "
              f"{r['bpb']:>7.4f} {r['learning_rate']:>9.1e} {r['weight_decay']:>5.1f}")


def main():
    parser = argparse.ArgumentParser(description="BPB conversion and best-run selection")
    parser.add_argument("--measure-ratio", action="store_true",
                        help="Measure bytes_per_token on C4 validation data")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of samples for ratio measurement")
    parser.add_argument("--ce-loss", type=float, default=None,
                        help="CE loss in nats to convert to BPB")
    parser.add_argument("--bytes-per-token", type=float, default=None,
                        help="Override bytes_per_token ratio")
    parser.add_argument("--best-runs", action="store_true",
                        help="Select best runs from dolma_val_loss sweeps")
    parser.add_argument("--size", type=str, default=None,
                        help="Filter by model size (e.g. 370M, 30M)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export best runs to JSON file")
    args = parser.parse_args()

    if args.measure_ratio:
        ratio = measure_bytes_per_token(args.num_samples)
        print(f"\nTo use this ratio:")
        print(f"  python compute_bpb.py --ce-loss <val> --bytes-per-token {ratio:.4f}")
        return

    if args.ce_loss is not None:
        bpt = args.bytes_per_token or DEFAULT_BYTES_PER_TOKEN
        bpb = ce_to_bpb(args.ce_loss, bpt)
        print(f"CE loss: {args.ce_loss:.4f} nats/token")
        print(f"Bytes/token: {bpt:.4f}")
        print(f"BPB: {bpb:.4f}")
        return

    if args.best_runs:
        records = select_best_runs(size_filter=args.size)
        print_best_runs(records)
        if args.export:
            with open(args.export, "w") as f:
                json.dump(records, f, indent=2)
            print(f"\nExported {len(records)} records to {args.export}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
