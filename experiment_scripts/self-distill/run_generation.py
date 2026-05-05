#!/usr/bin/env python3
"""
Orchestrator for self-distillation teacher generation.

Reads `teacher_manifest.json` keyed by (model_size, chinchilla_multiplier), resolves the
teacher checkpoint + original dolma shard size, auto-converts the teacher to HuggingFace
format if needed, then execs `generate_unconditional_vllm.py` to produce N_target files
of D tokens each (D = original dolma shard size on disk / 4).

Output layout (under the dolma data root, alongside paraphrase data):
    {data_root}/preprocessed/dolma2-0625/resharded/dolma2-tokenizer/self_distill/
        {basename(teacher_run_dir)}_{teacher_step}_temp{temperature}/
            output_000000.npy
            output_000001.npy
            ...

Usage:
    python run_generation.py --model-size 30M --chinchilla-multiplier 0.05
    python run_generation.py --model-size 30M --chinchilla-multiplier 0.05 --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# (model_size, chinchilla_multiplier) -> dolma shard basename (no .npy).
# Mirrors `_DATASET_LOOKUP` in src/scripts/official/OLMo-scale-train-paraphrase-dolma.py
# and OLMo-scale-train-multiepoch-dolma.py. Duplicated here so the orchestrator is a
# standalone script (no Python path setup required).
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

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # OLMo-core/
_HF_CONVERT_SCRIPT = _REPO_ROOT / "src/examples/huggingface/convert_checkpoint_to_hf.py"
_GENERATOR_SCRIPT = _HERE / "generate_unconditional_vllm.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", required=True,
                        help="e.g. 30M, 60M, 190M, 370M")
    parser.add_argument("--chinchilla-multiplier", required=True, type=float,
                        help="e.g. 0.05, 0.1, 1, 4")
    parser.add_argument("--data-root", default="/n/netscratch/barak_lab/Everyone/sqin/olmo",
                        help="Root containing preprocessed/dolma2-0625/...")
    parser.add_argument("--manifest", default=str(_HERE / "teacher_manifest.json"),
                        help="Path to teacher_manifest.json")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel SBATCH shards for this cell. "
                        "Submit N independent jobs with --shard-id 0..N-1. Each job "
                        "generates total_tokens/num_shards tokens.")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="0-indexed shard id for this job, in [0, num_shards)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved params without converting or generating")
    return parser.parse_args()


def lookup_manifest_entry(manifest_path: Path, model_size: str, chin: float) -> dict:
    with manifest_path.open() as f:
        manifest = json.load(f)
    chin_str = int(chin) if chin == int(chin) else chin
    key = f"{model_size}_chin{chin_str}"
    if key not in manifest:
        valid = [k for k in manifest if not k.startswith("_")]
        raise KeyError(
            f"No manifest entry for '{key}'. Add one to {manifest_path}. "
            f"Existing entries: {valid}"
        )
    return manifest[key]


def resolve_dolma_shard(data_root: Path, model_size: str, chin: float) -> Path:
    chin_key = int(chin) if chin == int(chin) else chin
    shard_basename = _DATASET_LOOKUP.get((model_size, chin_key))
    if shard_basename is None:
        valid = sorted(c for (m, c) in _DATASET_LOOKUP if m == model_size)
        raise KeyError(
            f"No dolma shard for ({model_size}, {chin}). "
            f"Valid chinchilla values for {model_size}: {valid}"
        )
    shard_path = data_root / _DOLMA_SUBPATH / f"{shard_basename}.npy"
    if not shard_path.exists():
        raise FileNotFoundError(f"Dolma shard not found: {shard_path}")
    return shard_path


def _patch_hf_config_for_vllm(hf_dir: Path, sequence_length: int) -> None:
    """
    Rewrite OLMo3 HF config to look like an OLMo2 config so vllm's native
    `olmo2.py` model class will load it. OLMo3 == OLMo2 + sliding-window
    attention; the architectures share identical weight tensors and key names
    (verified in src/olmo_core/nn/transformer/config.py: every olmo3_NN
    classmethod just calls olmo2_NN with sliding_window+attn_backend kwargs,
    no extra params). At seq_len <= window=4096, sliding attention is
    mathematically equivalent to full causal attention.

    Required edits:
      - architectures: ["Olmo2ForCausalLM"]
      - model_type: "olmo2"
      - drop sliding_window and layer_types (Olmo2Config doesn't know them)

    Idempotent: safe to call on already-patched configs.
    """
    config_path = hf_dir / "config.json"
    with config_path.open() as f:
        cfg = json.load(f)
    cfg_window = cfg.get("sliding_window")
    if cfg_window is not None and cfg_window < sequence_length:
        raise RuntimeError(
            f"HF config sliding_window={cfg_window} < sequence_length={sequence_length}; "
            f"stripping it would change semantics. Refusing to patch."
        )
    changed = False
    if cfg.get("architectures") != ["Olmo2ForCausalLM"]:
        cfg["architectures"] = ["Olmo2ForCausalLM"]
        changed = True
    if cfg.get("model_type") != "olmo2":
        cfg["model_type"] = "olmo2"
        changed = True
    for key in ("sliding_window", "layer_types"):
        if key in cfg:
            del cfg[key]
            changed = True
    if changed:
        with config_path.open("w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[run_generation] Patched HF config (olmo3 -> olmo2): {config_path}",
              flush=True)


def ensure_hf_checkpoint(teacher_run_dir: Path, teacher_step: str,
                         sequence_length: int, dry_run: bool) -> Path:
    step_dir = teacher_run_dir / teacher_step
    hf_dir = teacher_run_dir / f"{teacher_step}_hf"
    if not step_dir.exists():
        raise FileNotFoundError(f"Teacher step dir not found: {step_dir}")
    if (hf_dir / "config.json").exists():
        print(f"[run_generation] HF checkpoint already exists: {hf_dir}", flush=True)
        if not dry_run:
            _patch_hf_config_for_vllm(hf_dir, sequence_length)
        return hf_dir
    cmd = [
        sys.executable, str(_HF_CONVERT_SCRIPT),
        "-i", str(step_dir),
        "-o", str(hf_dir),
        "--max-sequence-length", str(sequence_length),
    ]
    print(f"[run_generation] Converting teacher to HF format:\n  {' '.join(cmd)}",
          flush=True)
    if dry_run:
        return hf_dir
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
    if not (hf_dir / "config.json").exists():
        raise RuntimeError(f"HF conversion did not produce config.json at {hf_dir}")
    _patch_hf_config_for_vllm(hf_dir, sequence_length)
    return hf_dir


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root)

    entry = lookup_manifest_entry(manifest_path, args.model_size,
                                  args.chinchilla_multiplier)
    teacher_run_dir = Path(entry["teacher_run_dir"])
    teacher_step = entry["teacher_step"]
    temperature = float(entry["temperature"])
    n_target = int(entry["N_target"])

    shard_path = resolve_dolma_shard(data_root, args.model_size,
                                     args.chinchilla_multiplier)
    d_tokens = shard_path.stat().st_size // 4  # uint32
    total_tokens = d_tokens * n_target

    out_dir_name = f"{teacher_run_dir.name}_{teacher_step}_temp{temperature}"
    output_dir = data_root / _DOLMA_SUBPATH / _SELF_DISTILL_SUBPATH / out_dir_name

    print(f"[run_generation] cell={args.model_size} chin={args.chinchilla_multiplier}",
          flush=True)
    print(f"[run_generation] dolma shard: {shard_path} (D={d_tokens:,} tokens)",
          flush=True)
    print(f"[run_generation] teacher: {teacher_run_dir.name}/{teacher_step} temp={temperature}",
          flush=True)
    print(f"[run_generation] output_dir: {output_dir}", flush=True)
    print(f"[run_generation] N_target={n_target}, total_tokens={total_tokens:,}",
          flush=True)

    hf_dir = ensure_hf_checkpoint(teacher_run_dir, teacher_step,
                                  args.sequence_length, args.dry_run)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # FlashAttention supports head_dim in {32, 64, 96, 128, 192, 256}.
    # OLMo head_dim by size:
    #   14M -> 16 (Flash NO -> FlexAttention)
    #   30M -> 32 (Flash YES)
    #   60M -> 48 (Flash NO -> FlexAttention)
    #   190M -> 64 (Flash YES)
    # FlexAttention's triton kernel cannot index past ~16M KV-cache tokens
    # and OOMs the (max_num_seqs * total_blocks) bookkeeping tensor on H100
    # at default settings. Cap both memory and concurrency for these sizes.
    is_flex = args.model_size in ("14M", "60M")
    gpu_mem_util = 0.20 if is_flex else 0.9

    cmd = [
        sys.executable, str(_GENERATOR_SCRIPT),
        "--model-path", str(hf_dir),
        "--output-dir", str(output_dir),
        "--total-tokens", str(total_tokens),
        "--sequence-length", str(args.sequence_length),
        "--batch-size", str(args.batch_size),
        "--temperature", str(temperature),
        "--tokens-per-file", str(d_tokens),
        "--seed", str(args.seed),
        "--num-shards", str(args.num_shards),
        "--shard-id", str(args.shard_id),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--resume",
    ]
    if is_flex:
        cmd += ["--max-num-seqs", "64"]
    print(f"[run_generation] Launching generator:\n  {' '.join(cmd)}", flush=True)
    if args.dry_run:
        print("[run_generation] --dry-run set; exiting without generation", flush=True)
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
