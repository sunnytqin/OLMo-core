"""
Training script for Case 4 (multi-scale): Extended DCLM baseline using dolma resharded data.

Supports multiple model scales via model_size= override. Data mix selection uses an explicit
lookup table so every (model_size, chinchilla_multiplier) pair is precisely defined — any
combination not in the table raises an error immediately.

Usage:
    torchrun ... OLMo-scale-train-case4-dclm-extended.py \
        --save-folder=... --data-root=... \
        model_size=30M chinchilla_multiplier=4 lr=1e-3 weight_decay=0.1

Data root should point to /n/netscratch/barak_lab/Everyone/sqin/olmo (dolma data).
"""

import argparse
from typing import List

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
# Model registry
# Maps size label -> (config_fn, approximate_param_count)
# approximate_param_count is used to compute chinchilla-optimal token budgets:
#   base_tokens = approx_params * 20
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    "30M":  (TransformerConfig.olmo3_30M,  30_000_000),
    "60M":  (TransformerConfig.olmo3_60M,  60_000_000),
    "190M": (TransformerConfig.olmo3_190M, 190_000_000),
    "370M": (TransformerConfig.olmo3_370M, 370_000_000),
}

# ---------------------------------------------------------------------------
# Explicit dataset lookup: (model_size, chinchilla_multiplier) -> DataMix
#
# Only combinations where model_params * 20 * chinchilla aligns exactly with
# an available dolma split are listed. Anything else raises an error.
#
# Token counts per entry (model_params * 20 * chin):
#   30M  * 20 * chin ->  600M * chin
#   60M  * 20 * chin -> 1200M * chin
#   190M * 20 * chin -> 3800M * chin
#   370M * 20 * chin -> 7400M * chin
# ---------------------------------------------------------------------------
_DATASET_LOOKUP = {
    # 30M model (600M tokens per chinchilla unit)
    ("30M", 0.05): DataMix.OLMo_dolma_0_03B,   #  30M tokens
    ("30M", 0.1):  DataMix.OLMo_dolma_0_06B,   #  60M tokens
    ("30M", 0.25): DataMix.OLMo_dolma_0_15B,   # 150M tokens
    ("30M", 0.5):  DataMix.OLMo_dolma_0_3B,    # 300M tokens
    ("30M", 1):    DataMix.OLMo_dolma_0_6B,    # 600M tokens
    ("30M", 2):    DataMix.OLMo_dolma_1_2B,    # 1.2B tokens
    ("30M", 4):    DataMix.OLMo_dolma_2_4B,    # 2.4B tokens
    ("30M", 8):    DataMix.OLMo_dolma_4_8B,    # 4.8B tokens
    ("30M", 16):   DataMix.OLMo_dolma_9_6B,    # 9.6B tokens
    # 60M model (1200M tokens per chinchilla unit)
    ("60M", 0.05): DataMix.OLMo_dolma_0_06B,   #  60M tokens
    ("60M", 0.1):  DataMix.OLMo_dolma_0_12B,   # 120M tokens
    ("60M", 0.25): DataMix.OLMo_dolma_0_3B,    # 300M tokens
    ("60M", 0.5):  DataMix.OLMo_dolma_0_6B,    # 600M tokens
    ("60M", 1):    DataMix.OLMo_dolma_1_2B,    # 1.2B tokens
    ("60M", 2):    DataMix.OLMo_dolma_2_4B,    # 2.4B tokens
    ("60M", 4):    DataMix.OLMo_dolma_4_8B,    # 4.8B tokens
    ("60M", 8):    DataMix.OLMo_dolma_9_6B,    # 9.6B tokens
    ("60M", 16):   DataMix.OLMo_dolma_19_2B,   # 19.2B tokens
    # 190M model (3800M tokens per chinchilla unit)
    ("190M", 0.05): DataMix.OLMo_dolma_0_19B,  #  190M tokens
    ("190M", 0.1):  DataMix.OLMo_dolma_0_38B,  #  380M tokens
    ("190M", 0.25): DataMix.OLMo_dolma_0_95B,  #  950M tokens
    ("190M", 0.5):  DataMix.OLMo_dolma_1_9B,   #  1.9B tokens
    ("190M", 1):    DataMix.OLMo_dolma_3_8B,   #  3.8B tokens
    ("190M", 2):    DataMix.OLMo_dolma_7_6B,   #  7.6B tokens
    ("190M", 4):    DataMix.OLMo_dolma_15_2B,  # 15.2B tokens
    ("190M", 8):    DataMix.OLMo_dolma_30_4B,  # 30.4B tokens
    ("190M", 16):   DataMix.OLMo_dolma_60_8B,  # 60.8B tokens
    # 370M model (7400M tokens per chinchilla unit)
    ("370M", 0.05): DataMix.OLMo_dolma_0_37B,  #  370M tokens
    ("370M", 0.1):  DataMix.OLMo_dolma_0_74B,  #  740M tokens
    ("370M", 0.25): DataMix.OLMo_dolma_1_85B,  # 1.85B tokens
    ("370M", 0.5):  DataMix.OLMo_dolma_3_7B,   #  3.7B tokens
    ("370M", 1):    DataMix.OLMo_dolma_7_4B,   #  7.4B tokens
    ("370M", 2):    DataMix.OLMo_dolma_14_8B,  # 14.8B tokens
    ("370M", 4):    DataMix.OLMo_dolma_29_6B,  # 29.6B tokens
    ("370M", 8):    DataMix.OLMo_dolma_59_2B,  # 59.2B tokens
}


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    # Defaults
    init_seed = 42
    lr = 1e-3
    weight_decay = 0.1
    chinchilla_multiplier = 1.0
    epochs = 1
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
        elif override.startswith("epochs="):
            epochs = float(override.split("=")[1])
        elif override.startswith("model_size="):
            model_size = override.split("=")[1]
        elif override.startswith("microbatch_multiplier="):
            microbatch_multiplier = int(override.split("=")[1])
        elif override.startswith("eval_only="):
            eval_only = override.split("=")[1].lower() in ("true", "1")
        else:
            filtered_overrides.append(override)

    # Validate model size
    if model_size not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_size={model_size!r}. Available: {sorted(_MODEL_REGISTRY)}"
        )
    config_fn, model_params = _MODEL_REGISTRY[model_size]

    # Look up dataset — strict: only explicitly listed (model_size, chinchilla) pairs allowed
    lookup_key = (model_size, chinchilla_multiplier)
    if lookup_key not in _DATASET_LOOKUP:
        valid = sorted(k for k in _DATASET_LOOKUP if k[0] == model_size)
        raise ValueError(
            f"No dolma data mix defined for model_size={model_size}, "
            f"chinchilla_multiplier={chinchilla_multiplier}. "
            f"Valid chinchilla values for {model_size}: {[k[1] for k in valid]}"
        )
    data_mix = _DATASET_LOOKUP[lookup_key]

    # Token budget: model_params * 20 per chinchilla unit, times epochs
    base_tokens_per_epoch = model_params * 20
    total_training_tokens = int(base_tokens_per_epoch * chinchilla_multiplier * epochs)

    # Warmup: 2% of total training steps
    global_batch_size = 512 * 4096
    total_steps = total_training_tokens // global_batch_size
    warmup_steps = max(1, int(total_steps * 0.02))

    # Run name
    chin_str = int(chinchilla_multiplier) if chinchilla_multiplier == int(chinchilla_multiplier) else chinchilla_multiplier
    epoch_str = int(epochs) if epochs == int(epochs) else epochs
    run_name = f"{model_size}_seed{init_seed:02d}_multiepoch_dolma_chin{chin_str}_epoch{epoch_str}_wd{weight_decay}_lr{lr}"

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = config_fn(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        data_mix,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
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
