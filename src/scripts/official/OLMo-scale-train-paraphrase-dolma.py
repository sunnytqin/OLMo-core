"""
Training script for D + D' paraphrase scaling: train one epoch on the original
dolma shard `D` concatenated with K paraphrase seeds (`D'`) of the same docs.

Sweep axes (parallel of OLMo-scale-train-multiepoch-dolma.py):
    model_size, chinchilla_multiplier, num_seeds (K, in 1..16)

The paraphrased data is built by experiment_scripts/paraphrasing/build_sized_paraphrase.py,
which produces paraphrased/sized_smollm2_mixed/{shard}_seed{N}.npy aligned doc-by-doc
with the original {shard}.npy.

Paraphrased docs are NOT length-preserving (typically much shorter than originals,
sometimes 30% the token count). So `total_training_tokens` is computed by summing
the actual on-disk byte sizes of all source files, not from a chinchilla formula.

Usage:
    torchrun ... OLMo-scale-train-paraphrase-dolma.py \
        --save-folder=... --data-root=... \
        model_size=30M chinchilla_multiplier=1 num_seeds=2 lr=1e-3 weight_decay=0.1

Data root should point to /n/netscratch/barak_lab/Everyone/sqin/olmo (dolma data).
Only `(model_size, chinchilla_multiplier)` combos with `D <= 7.4B` are supported,
because the paraphrase corpus only covers the 4,866,732 docs of train_7.4B.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# Model registry — same as multi-epoch script.
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    "14M":  (TransformerConfig.olmo3_14M,   14_000_000),
    "30M":  (TransformerConfig.olmo3_30M,   30_000_000),
    "60M":  (TransformerConfig.olmo3_60M,   60_000_000),
    "100M": (TransformerConfig.olmo3_100M, 100_000_000),
    "190M": (TransformerConfig.olmo3_190M, 190_000_000),
    "370M": (TransformerConfig.olmo3_370M, 370_000_000),
    "600M": (TransformerConfig.olmo3_600M, 600_000_000),
}

# ---------------------------------------------------------------------------
# (model_size, chinchilla_multiplier) -> dolma shard basename (no .npy).
#
# Truncated where D would exceed 7.4B tokens — the paraphrase corpus only
# covers the 4,866,732 docs of train_7.4B. Combos not listed here raise.
# Supported K ∈ {1..16}.
# ---------------------------------------------------------------------------
_DATASET_LOOKUP = {
    # 14M (chin unit = 280M tokens)
    ("14M", 0.05): "train_0.014B",
    ("14M", 0.1):  "train_0.028B",
    ("14M", 0.25): "train_0.07B",
    ("14M", 0.5):  "train_0.14B",
    ("14M", 1):    "train_0.28B",
    ("14M", 2):    "train_0.56B",
    ("14M", 4):    "train_1.12B",
    ("14M", 8):    "train_2.24B",
    ("14M", 16):   "train_4.48B",
    # 30M (chin unit = 600M tokens)
    ("30M", 0.05): "train_0.03B",
    ("30M", 0.1):  "train_0.06B",
    ("30M", 0.25): "train_0.15B",
    ("30M", 0.5):  "train_0.3B",
    ("30M", 1):    "train_0.6B",
    ("30M", 2):    "train_1.2B",
    ("30M", 4):    "train_2.4B",
    ("30M", 8):    "train_4.8B",
    # 60M (chin unit = 1.2B tokens)
    ("60M", 0.05): "train_0.06B",
    ("60M", 0.1):  "train_0.12B",
    ("60M", 0.25): "train_0.3B",
    ("60M", 0.5):  "train_0.6B",
    ("60M", 1):    "train_1.2B",
    ("60M", 2):    "train_2.4B",
    ("60M", 4):    "train_4.8B",
    # 100M (chin unit = 2.0B tokens)
    ("100M", 0.05): "train_0.1B",
    ("100M", 0.1):  "train_0.2B",
    ("100M", 0.25): "train_0.5B",
    ("100M", 0.5):  "train_1.0B",
    ("100M", 1):    "train_2.0B",
    ("100M", 2):    "train_4.0B",
    # 190M (chin unit = 3.8B tokens)
    ("190M", 0.05): "train_0.19B",
    ("190M", 0.1):  "train_0.38B",
    ("190M", 0.25): "train_0.95B",
    ("190M", 0.5):  "train_1.9B",
    ("190M", 1):    "train_3.8B",
    ("190M", 2):    "train_7.4B",
    # 370M (chin unit = 7.4B tokens)
    ("370M", 0.05): "train_0.37B",
    ("370M", 0.1):  "train_0.74B",
    ("370M", 0.25): "train_1.85B",
    ("370M", 0.5):  "train_3.7B",
    ("370M", 1):    "train_7.4B",
    # 600M (chin unit = 12.0B tokens)
    ("600M", 0.05): "train_0.6B",
    ("600M", 0.1):  "train_1.2B",
    ("600M", 0.25): "train_3.0B",
    ("600M", 0.5):  "train_6.0B",
}

_DOLMA_SUBPATH = "preprocessed/dolma2-0625/resharded"
_PARAPHRASE_SUBPATH = "paraphrased/sized_smollm2_mixed"


def _build_dataset_paths(
    data_root: str,
    tokenizer_id: str,
    shard_name: str,
    num_seeds: int,
) -> Tuple[List[str], List[dict]]:
    """
    Construct the concrete file paths for D + D'_seed1 + ... + D'_seedK and
    matching per-source metadata labels.
    """
    base = f"{data_root.rstrip('/')}/{_DOLMA_SUBPATH}/{tokenizer_id}"
    original = f"{base}/{shard_name}.npy"
    paraphrased = [
        f"{base}/{_PARAPHRASE_SUBPATH}/{shard_name}_seed{k}.npy"
        for k in range(1, num_seeds + 1)
    ]
    paths = [original] + paraphrased
    metadata = [{"label": "dolma"}] + [
        {"label": f"dolma_para_seed{k}"} for k in range(1, num_seeds + 1)
    ]
    return paths, metadata


def _sum_tokens_on_disk(paths: List[str]) -> int:
    """Sum the on-disk uint32 token counts across all source files."""
    total = 0
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                f"Run experiment_scripts/paraphrasing/build_sized_paraphrase.py "
                f"to materialize sized paraphrase shards."
            )
        total += path.stat().st_size // 4  # uint32 = 4 bytes per token
    return total


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    # Defaults
    init_seed = 42
    lr = 1e-3
    weight_decay = 0.1
    chinchilla_multiplier = 1.0
    num_seeds = 1
    model_size = "30M"
    microbatch_multiplier = 16
    eval_only = False
    sequence_length = opts.sequence_length if opts.sequence_length is not None else 4096

    filtered_overrides = []
    for override in overrides:
        if override.startswith("init_seed="):
            init_seed = int(override.split("=")[1])
        elif override.startswith("lr="):
            lr = float(override.split("=")[1])
        elif override.startswith("weight_decay="):
            weight_decay = float(override.split("=")[1])
        elif override.startswith("chinchilla_multiplier="):
            chinchilla_multiplier = float(override.split("=")[1])
        elif override.startswith("num_seeds="):
            num_seeds = int(override.split("=")[1])
        elif override.startswith("model_size="):
            model_size = override.split("=")[1]
        elif override.startswith("microbatch_multiplier="):
            microbatch_multiplier = int(override.split("=")[1])
        elif override.startswith("eval_only="):
            eval_only = override.split("=")[1].lower() in ("true", "1")
        else:
            filtered_overrides.append(override)

    if model_size not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_size={model_size!r}. Available: {sorted(_MODEL_REGISTRY)}"
        )
    config_fn, model_params = _MODEL_REGISTRY[model_size]

    if not 1 <= num_seeds <= 16:
        raise ValueError(
            f"num_seeds must be in 1..16 (only 16 V2 paraphrase seeds exist), "
            f"got {num_seeds}. For K=0 baselines use OLMo-scale-train-multiepoch-dolma.py."
        )

    lookup_key = (model_size, chinchilla_multiplier)
    if lookup_key not in _DATASET_LOOKUP:
        valid = sorted(k[1] for k in _DATASET_LOOKUP if k[0] == model_size)
        raise ValueError(
            f"No paraphrase dataset for model_size={model_size}, "
            f"chinchilla_multiplier={chinchilla_multiplier}. "
            f"Valid chinchilla values for {model_size}: {valid}. "
            f"(Combos exceeding D=7.4B are excluded — paraphrase corpus only "
            f"covers train_7.4B's 4.87M docs.)"
        )
    shard_name = _DATASET_LOOKUP[lookup_key]

    tokenizer_config = TokenizerConfig.dolma2()
    if tokenizer_config.identifier is None:
        raise RuntimeError("dolma2 tokenizer has no identifier; cannot resolve paths.")

    paths, metadata = _build_dataset_paths(
        data_root=opts.data_root,
        tokenizer_id=tokenizer_config.identifier,
        shard_name=shard_name,
        num_seeds=num_seeds,
    )

    # Token budget: actual on-disk size, not a (1+K)*|D| approximation.
    total_training_tokens = _sum_tokens_on_disk(paths)

    naive_estimate = int(model_params * 20 * chinchilla_multiplier * (1 + num_seeds))
    drift_pct = (total_training_tokens - naive_estimate) / max(naive_estimate, 1) * 100
    print(
        f"[paraphrase-dolma] model={model_size} chin={chinchilla_multiplier} K={num_seeds} "
        f"shard={shard_name}",
        flush=True,
    )
    print(
        f"[paraphrase-dolma] total_training_tokens (on-disk) = {total_training_tokens:,}",
        flush=True,
    )
    print(
        f"[paraphrase-dolma] naive (1+K)*chin*params*20 estimate = {naive_estimate:,} "
        f"(drift {drift_pct:+.1f}%)",
        flush=True,
    )

    global_batch_size = 512 * 4096
    total_steps = total_training_tokens // global_batch_size
    warmup_steps = max(1, int(total_steps * 0.02))

    chin_str = (
        int(chinchilla_multiplier)
        if chinchilla_multiplier == int(chinchilla_multiplier)
        else chinchilla_multiplier
    )
    run_name = (
        f"{model_size}_seed{init_seed:02d}_dolma_para_chin{chin_str}_K{num_seeds}"
        f"_wd{weight_decay}_lr{lr}"
    )

    model_config = config_fn(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    dataset_config = NumpyFSLDatasetConfig(
        paths=paths,
        metadata=metadata,
        tokenizer=tokenizer_config,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=init_seed,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=microbatch_multiplier * 4096,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=True,
        ),
        scheduler=CosWithWarmup(warmup_steps=warmup_steps),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp if get_world_size() == 1 else DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=max(1, get_world_size() // 2) if get_world_size() > 1 else 1,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            TransformerActivationCheckpointingMode.full
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    if get_world_size() >= 1024:
        train_module_config.rank_microbatch_size //= 2
        train_module_config.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(total_training_tokens),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=2000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity="harvardml",
                project="syn_data_scaling",
                cancel_check_interval=10,
                enabled=True,
                tags=["eval"] if eval_only else ["train"],
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.OLMo_dolma_val,
                    mix_base_dir=opts.data_root,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                eval_interval=min(200, max(1, total_steps // 2)),
                eval_duration=Duration.tokens(50_000_000),
                eval_on_startup=eval_only,
                cancel_after_first_eval=eval_only,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=init_seed,
    ).merge(filtered_overrides)


if __name__ == "__main__":
    main(build_config)
