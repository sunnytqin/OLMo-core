"""
Training script for Case 4: Extended DCLM (training on fresh data always).
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
    epochs = 1  # Default number of epochs
    dclm_repeat_factor = 1  # Not used in Case 4, but accepted for compatibility with bash script
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
            chinchilla_multiplier = float(override.split("=")[1])
        elif override.startswith("epochs="):
            epochs = float(override.split("=")[1])
        elif override.startswith("dclm_repeat_factor="):
            dclm_repeat_factor = int(override.split("=")[1])  # Accepted but not used in Case 4
        elif override.startswith("microbatch_multiplier="):
            microbatch_multiplier = int(override.split("=")[1])
        else:
            # Keep other overrides for ExperimentConfig
            filtered_overrides.append(override)

    # Compute training parameters based on chinchilla_multiplier and epochs
    # Base: 0.6B tokens per epoch of DCLM data
    # For multi-epoch training: total_tokens = base * chinchilla_multiplier * epochs
    # Dataset selection is still based on chinchilla_multiplier only
    base_tokens_per_epoch = 0.6e9
    total_training_tokens = int(base_tokens_per_epoch * chinchilla_multiplier * epochs)

    # Compute warmup steps (~1.31% of total training, based on previous experiments)
    # With global_batch_size=512*4096=2,097,152 tokens per step
    # Previous: chin_4 used 14 steps (1.22%), chin_16 used 64 steps (1.40%)
    global_batch_size = 512 * 4096
    total_steps = total_training_tokens // global_batch_size
    warmup_ratio = 0.013107
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    # Auto-generate run name based on model size, seed, and hyperparameters
    # Format chinchilla_multiplier and epochs to avoid ".0" for whole numbers
    chin_str = int(chinchilla_multiplier) if chinchilla_multiplier == int(chinchilla_multiplier) else chinchilla_multiplier
    epoch_str = int(epochs) if epochs == int(epochs) else epochs
    run_name = f"30M_seed{init_seed:02d}_case4_dclm_extended_chin{chin_str}_epoch{epoch_str}_wd{weight_decay}_lr{lr}"

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo2_30M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    # Select the appropriate extended dataset based on chinchilla_multiplier
    dataset_mix_map = {
        0.5: DataMix.OLMo_dclm_chin0_5,
        1: DataMix.OLMo_dclm_chin1,
        2: DataMix.OLMo_dclm_chin2,
        4: DataMix.OLMo_dclm_chin4,
        8: DataMix.OLMo_dclm_chin8,
        16: DataMix.OLMo_dclm_chin16,
    }
    if chinchilla_multiplier not in dataset_mix_map:
        raise ValueError(f"No extended dataset available for chinchilla_multiplier={chinchilla_multiplier}. Available: {list(dataset_mix_map.keys())}")

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        dataset_mix_map[chinchilla_multiplier],
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
                save_interval=2000,
                ephemeral_save_interval=200,
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
