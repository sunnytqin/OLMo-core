"""
Training script for Case 2: DCLM + Synthetic (Chinchilla-scaled, no repetition).
Supports chin4 (2.4B), chin8 (4.8B), and chin16 (9.6B) training tokens.
Can optionally repeat DCLM data with dclm_repeat_factor parameter (e.g., dclm_repeat_factor=4).
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
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    # Training seed - can be overridden via command line argument
    # Check if init_seed is provided in overrides, otherwise use default
    init_seed = 42  # Default seed
    lr = 1e-3  # Default learning rate
    weight_decay = 0.1  # Default weight decay
    chinchilla_multiplier = 4  # Default chinchilla multiplier (e.g., 4 means 4x0.6B=2.4B tokens)
    dclm_repeat_factor = 1  # Default DCLM repeat factor (1 = no repeat, 4 = repeat DCLM 4 times)
    epochs = 1  # Not used in Case 2, but accepted for compatibility with bash script
    microbatch_multiplier = 16  # Default microbatch multiplier (rank_microbatch_size = microbatch_multiplier * 4096)

    # Filter out hyperparameters that are not part of ExperimentConfig
    filtered_overrides = []
    for override in overrides:
        if override.startswith("init_seed="):
            init_seed = int(override.split("=")[1])
        elif override.startswith("lr="):
            lr = float(override.split("=")[1])
        elif override.startswith("weight_decay="):
            weight_decay = float(override.split("=")[1])
        elif override.startswith("chinchilla_multiplier="):
            chinchilla_multiplier = int(override.split("=")[1])
        elif override.startswith("dclm_repeat_factor="):
            dclm_repeat_factor = int(override.split("=")[1])
        elif override.startswith("epochs="):
            epochs = int(override.split("=")[1])  # Accepted but not used in Case 2
        elif override.startswith("microbatch_multiplier="):
            microbatch_multiplier = int(override.split("=")[1])
        else:
            # Keep other overrides for ExperimentConfig
            filtered_overrides.append(override)

    # Compute training parameters based on chinchilla_multiplier
    # Base: 0.6B tokens per epoch of DCLM data
    base_tokens_per_epoch = 0.3e9
    total_training_tokens = int(base_tokens_per_epoch * chinchilla_multiplier)

    # Compute warmup steps (~1.31% of total training, based on previous experiments)
    global_batch_size = 64 * 4096
    total_steps = total_training_tokens // global_batch_size
    warmup_ratio = 0.013107
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    # Auto-generate run name based on model size, seed, and hyperparameters
    repeat_suffix = f"_repeat{dclm_repeat_factor}" if dclm_repeat_factor > 1 else ""
    run_name = f"370M_seed{init_seed:02d}_case2_dclm_synthetic_chin{chinchilla_multiplier}{repeat_suffix}_wd{weight_decay}_lr{lr}"

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo2_370M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    # Select the appropriate synthetic dataset based on dclm_repeat_factor
    # For newer datasets (repeat32/repeat64), selection is based only on dclm_repeat_factor
    # For older datasets, selection is based on both chinchilla_multiplier and dclm_repeat_factor
    if dclm_repeat_factor == 32:
        data_mix = DataMix.OLMo_repeat32_synthetic32
    elif dclm_repeat_factor == 64:
        data_mix = DataMix.OLMo_repeat64_synthetic32
    else:
        # Fallback to old tuple-based lookup for chin4, chin8, chin16 datasets
        dataset_mix_map = {
            (4, 1): DataMix.OLMo_synthetic_chin4,
            (8, 1): DataMix.OLMo_synthetic_chin8,
            (16, 1): DataMix.OLMo_synthetic_chin16,
            (16, 4): DataMix.OLMo_synthetic_chin16_repeat4,
            (48, 16): DataMix.OLMo_repeat16_synthetic48,
        }
        mix_key = (chinchilla_multiplier, dclm_repeat_factor)
        if mix_key not in dataset_mix_map:
            raise ValueError(f"No synthetic dataset available for chinchilla_multiplier={chinchilla_multiplier} and dclm_repeat_factor={dclm_repeat_factor}. Available: {list(dataset_mix_map.keys())} or use dclm_repeat_factor=32 or 64 for newer datasets.")
        data_mix = dataset_mix_map[mix_key]

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        data_mix,
        tokenizer=tokenizer_config,
        mix_base_dir="/n/netscratch/dam_lab/Lab/sqin/olmo",
        sequence_length=opts.sequence_length,
        max_target_sequence_length=max(8192, opts.sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=init_seed,  # Use same seed for data shuffling
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=microbatch_multiplier * 4096,
        max_sequence_length=opts.sequence_length,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=True,
        ),
        scheduler=CosWithWarmup(warmup_steps=warmup_steps),  # ~1.31% of total training steps
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp if get_world_size() == 1 else DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=max(1, get_world_size() // 2) if get_world_size() > 1 else 1,  # Use FSDP for single GPU, HSDP for multi-GPU
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            TransformerActivationCheckpointingMode.full
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # If you have 1024 GPUs, you can run slightly faster with a different config.
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
            max_duration=Duration.tokens(total_training_tokens),  # Computed based on chinchilla_multiplier
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
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
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.dclm_validation,
                    mix_base_dir="/n/netscratch/dam_lab/Lab/sqin/olmo",
                    sequence_length=opts.sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                eval_interval=100,
                eval_duration=Duration.tokens(50_000_000),  # Evaluate on 50M tokens
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
