"""
Training script for self-distillation: train one student over a 1:1 mix of
(original dolma shard D, repeated K times) + (K teacher-generated synthetic files,
each ~|D| tokens, used once).

Sweep axes: model_size, chinchilla_multiplier, K (1..N_target_on_disk).

Teacher selection is read from experiment_scripts/self-distill/teacher_manifest.json
keyed by (model_size, chinchilla_multiplier). The synthetic-data directory is computed
from the teacher run name + step + temperature, and is expected to live at
    {data_root}/preprocessed/dolma2-0625/resharded/dolma2-tokenizer/self_distill/
        {basename(teacher_run_dir)}_{teacher_step}_temp{temperature}/

Usage:
    torchrun ... OLMo-scale-train-selfdistill-dolma.py \
        --save-folder=... --data-root=... \
        model_size=30M chinchilla_multiplier=0.05 K=2 lr=1e-3 weight_decay=0.1
"""

import argparse
import json
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
# Model registry — same as multi-epoch / paraphrase scripts.
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

# (model_size, chinchilla_multiplier) -> dolma shard basename (no .npy).
# Mirrors paraphrase script. Truncated where D would exceed 7.4B (paraphrase corpus
# limit) — the same bound is fine here since teacher data is bounded by what we generate
# anyway.
_DATASET_LOOKUP = {
    ("14M", 0.05): "train_0.014B", ("14M", 0.1): "train_0.028B",
    ("14M", 0.25): "train_0.07B", ("14M", 0.5): "train_0.14B",
    ("14M", 1): "train_0.28B", ("14M", 2): "train_0.56B",
    ("14M", 4): "train_1.12B", ("14M", 8): "train_2.24B",
    ("14M", 16): "train_4.48B",
    ("30M", 0.05): "train_0.03B", ("30M", 0.1): "train_0.06B",
    ("30M", 0.25): "train_0.15B", ("30M", 0.5): "train_0.3B",
    ("30M", 1): "train_0.6B", ("30M", 2): "train_1.2B",
    ("30M", 4): "train_2.4B", ("30M", 8): "train_4.8B",
    ("60M", 0.05): "train_0.06B", ("60M", 0.1): "train_0.12B",
    ("60M", 0.25): "train_0.3B", ("60M", 0.5): "train_0.6B",
    ("60M", 1): "train_1.2B", ("60M", 2): "train_2.4B",
    ("60M", 4): "train_4.8B",
    ("100M", 0.05): "train_0.1B", ("100M", 0.1): "train_0.2B",
    ("100M", 0.25): "train_0.5B", ("100M", 0.5): "train_1.0B",
    ("100M", 1): "train_2.0B", ("100M", 2): "train_4.0B",
    ("190M", 0.05): "train_0.19B", ("190M", 0.1): "train_0.38B",
    ("190M", 0.25): "train_0.95B", ("190M", 0.5): "train_1.9B",
    ("190M", 1): "train_3.8B", ("190M", 2): "train_7.4B",
    ("370M", 0.05): "train_0.37B", ("370M", 0.1): "train_0.74B",
    ("370M", 0.25): "train_1.85B", ("370M", 0.5): "train_3.7B",
    ("370M", 1): "train_7.4B",
    ("600M", 0.05): "train_0.6B", ("600M", 0.1): "train_1.2B",
    ("600M", 0.25): "train_3.0B", ("600M", 0.5): "train_6.0B",
}

_DOLMA_SUBPATH = "preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer"
_SELF_DISTILL_SUBPATH = "self_distill"

# Manifest lives under experiment_scripts/self-distill, sibling to this repo's
# experiment_scripts directory. Resolve relative to this file.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # OLMo-core/
_MANIFEST_PATH = _REPO_ROOT / "experiment_scripts/self-distill/teacher_manifest.json"


def _lookup_teacher_entry(model_size: str, chin: float) -> dict:
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Teacher manifest not found: {_MANIFEST_PATH}")
    with _MANIFEST_PATH.open() as f:
        manifest = json.load(f)
    chin_key = int(chin) if chin == int(chin) else chin
    key = f"{model_size}_chin{chin_key}"
    if key not in manifest:
        valid = [k for k in manifest if not k.startswith("_")]
        raise KeyError(
            f"No manifest entry for '{key}'. Add one to {_MANIFEST_PATH}. "
            f"Existing entries: {valid}"
        )
    return manifest[key]


def _teacher_output_dirname(entry: dict) -> str:
    teacher_basename = Path(entry["teacher_run_dir"]).name
    return f"{teacher_basename}_{entry['teacher_step']}_temp{float(entry['temperature'])}"


def _build_dataset_paths(
    data_root: str,
    shard_name: str,
    teacher_dirname: str,
    K: int,
) -> Tuple[List[str], List[dict]]:
    base = f"{data_root.rstrip('/')}/{_DOLMA_SUBPATH}"
    original_path = f"{base}/{shard_name}.npy"

    syn_dir = Path(base) / _SELF_DISTILL_SUBPATH / teacher_dirname
    if not syn_dir.exists():
        raise FileNotFoundError(
            f"Synthetic-data dir not found: {syn_dir}\n"
            f"Run experiment_scripts/self-distill/submit_generation.sh first."
        )
    syn_files = sorted(syn_dir.glob("output_*.npy"))
    if len(syn_files) < K:
        raise FileNotFoundError(
            f"Need K={K} synthetic files; only {len(syn_files)} available in {syn_dir}. "
            f"Either reduce K or finish generation."
        )

    paths = [original_path] * K + [str(p) for p in syn_files[:K]]
    metadata = (
        [{"label": "dolma"}] * K
        + [{"label": f"self_distill_{i}"} for i in range(K)]
    )
    return paths, metadata


def _sum_tokens_on_disk(paths: List[str]) -> int:
    total = 0
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
        total += path.stat().st_size // 4
    return total


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    init_seed = 42
    lr = 1e-3
    weight_decay = 0.1
    chinchilla_multiplier = 1.0
    K = 1
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
        elif override.startswith("K="):
            K = int(override.split("=")[1])
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

    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    chin_key = int(chinchilla_multiplier) if chinchilla_multiplier == int(chinchilla_multiplier) else chinchilla_multiplier
    if (model_size, chin_key) not in _DATASET_LOOKUP:
        valid = sorted(c for (m, c) in _DATASET_LOOKUP if m == model_size)
        raise ValueError(
            f"No dolma shard for model_size={model_size}, "
            f"chinchilla_multiplier={chinchilla_multiplier}. Valid: {valid}"
        )
    shard_name = _DATASET_LOOKUP[(model_size, chin_key)]

    entry = _lookup_teacher_entry(model_size, chinchilla_multiplier)
    teacher_dirname = _teacher_output_dirname(entry)

    paths, metadata = _build_dataset_paths(
        data_root=opts.data_root,
        shard_name=shard_name,
        teacher_dirname=teacher_dirname,
        K=K,
    )

    total_training_tokens = _sum_tokens_on_disk(paths)
    naive_estimate = int(model_params * 20 * chinchilla_multiplier * 2 * K)
    drift_pct = (total_training_tokens - naive_estimate) / max(naive_estimate, 1) * 100
    print(
        f"[selfdistill-dolma] model={model_size} chin={chinchilla_multiplier} K={K} "
        f"shard={shard_name} teacher={entry['teacher_run_dir']}/{entry['teacher_step']}",
        flush=True,
    )
    print(
        f"[selfdistill-dolma] total_training_tokens (on-disk) = {total_training_tokens:,}",
        flush=True,
    )
    print(
        f"[selfdistill-dolma] naive 2*K*chin*params*20 estimate = {naive_estimate:,} "
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
        f"{model_size}_seed{init_seed:02d}_dolma_selfdistill_chin{chin_str}_K{K}"
        f"_wd{weight_decay}_lr{lr}"
    )

    tokenizer_config = TokenizerConfig.dolma2()

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
                tags=["eval"] if eval_only else ["selfdistill"],
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
