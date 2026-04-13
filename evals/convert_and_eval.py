"""
Convert OLMo-core distributed checkpoints to HuggingFace format and run lm_eval.

Usage:
    python convert_and_eval.py --checkpoint /path/to/model/stepN --output-dir /tmp/hf_models --limit 5
    python convert_and_eval.py --all --output-dir /tmp/hf_models
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer import TransformerConfig

SHARED_DIR = Path("/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints/chinchilla_0.1/best_runs")
TASK_YAML_DIR = Path(__file__).parent
EVAL_TASKS = "syn_pt_eval"

# wandb project info
WANDB_ENTITY = "harvardml"
WANDB_PROJECT = "syn_data_scaling"


def get_all_checkpoints():
    """Discover all model checkpoints in shared/."""
    checkpoints = []
    for model_dir in sorted(SHARED_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == ".":
            continue
        for step_dir in sorted(model_dir.iterdir()):
            if step_dir.name.startswith("step") and (step_dir / "model_and_optim").exists():
                checkpoints.append(step_dir)
    return checkpoints


def convert_checkpoint(checkpoint_path: Path, output_dir: Path) -> Path:
    """Convert a single OLMo-core checkpoint to HuggingFace format."""
    model_name = checkpoint_path.parent.name
    step_name = checkpoint_path.name
    hf_path = output_dir / f"{model_name}_{step_name}_hf"

    if (hf_path / "config.json").exists():
        print(f"  [skip] HF checkpoint already exists: {hf_path}")
        return hf_path

    print(f"  Converting {model_name}/{step_name} -> {hf_path}")

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmo3_370M(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    # Override attention backend for CPU conversion
    from olmo_core.nn.attention import AttentionBackendName
    model_config.block.attention.backend = AttentionBackendName.torch
    model_config.block.attention.use_flash = False

    model = model_config.build(init_device="meta")
    model.to_empty(device=torch.device("cpu"))

    vocab_size = tokenizer_config.vocab_size
    tokenizer_id = tokenizer_config.identifier

    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    hf_path.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(hf_path)

    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = str(checkpoint_path / "model_and_optim")
        print(f"  Loading weights from {model_and_optim_dir}")
        load_model_and_optim_state(model_and_optim_dir, model, work_dir=work_dir)

        state_dict_options = dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

        print(f"  Saving HF checkpoint to {hf_path}")
        save_hf_model(
            hf_path,
            model_state_dict,
            model,
            huggingface_tokenizer=hf_tokenizer,
            vocab_size=vocab_size,
            work_dir=work_dir,
            save_overwrite=True,
        )

    # Fix config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(hf_path)
    hf_config.max_position_embeddings = 4096
    hf_config.pad_token_id = tokenizer_config.pad_token_id
    hf_config.bos_token_id = tokenizer_config.bos_token_id
    hf_config.eos_token_id = tokenizer_config.eos_token_id
    hf_config.save_pretrained(hf_path)

    print(f"  [done] Saved to {hf_path}")
    return hf_path


def run_eval(hf_path: Path, run_name: str, limit: int = None, output_dir: Path = None):
    """Run lm_eval on a HuggingFace checkpoint."""
    results_dir = output_dir or (TASK_YAML_DIR / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / run_name

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={hf_path},dtype=bfloat16,max_length=4096",
        "--include_path", str(TASK_YAML_DIR),
        "--tasks", EVAL_TASKS,
        "--batch_size", "auto",
        "--output_path", str(output_path),
    ]
    # Only add wandb logging if WANDB_LOG env var is set
    if os.environ.get("WANDB_LOG", "0") == "1":
        cmd.extend(["--wandb_args", f"project={WANDB_PROJECT},entity={WANDB_ENTITY},name={run_name},tags=eval"])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"\n{'='*60}")
    print(f"Running eval: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [ERROR] Eval failed for {run_name} (exit code {result.returncode})")
        return False
    print(f"  [done] Results saved to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert OLMo checkpoints and run lm_eval")
    parser.add_argument("--checkpoint", type=str, help="Path to a single checkpoint (e.g. shared/model/stepN)")
    parser.add_argument("--all", action="store_true", help="Run on all checkpoints in SHARED")
    parser.add_argument("--output-dir", type=str, default="/tmp/hf_models", help="Where to save HF checkpoints")
    parser.add_argument("--results-dir", type=str, default=None, help="Where to save eval results")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (for testing)")
    parser.add_argument("--convert-only", action="store_true", help="Only convert, don't eval")
    parser.add_argument("--eval-only", action="store_true", help="Only eval (assumes HF checkpoints exist)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) if args.results_dir else TASK_YAML_DIR / "results"

    if args.all:
        checkpoints = get_all_checkpoints()
    elif args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        parser.error("Specify --checkpoint or --all")

    print(f"Found {len(checkpoints)} checkpoint(s)")

    for ckpt in checkpoints:
        model_name = ckpt.parent.name
        step_name = ckpt.name
        run_name = f"{model_name}_{step_name}"
        hf_path = output_dir / f"{run_name}_hf"

        if not args.eval_only:
            hf_path = convert_checkpoint(ckpt, output_dir)

        if not args.convert_only:
            run_eval(hf_path, run_name, limit=args.limit, output_dir=results_dir)


if __name__ == "__main__":
    main()
