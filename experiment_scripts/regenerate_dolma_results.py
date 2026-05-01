#!/usr/bin/env python3
"""
Regenerate results/dolma_<size>.py from eval JSONs.

Reads:
  - results/dolma_val_loss/{chin}/{size}/*.json       (multi-epoch)
  - results/dolma_para_val_loss/{chin}/{size}/*.json  (paraphrase)
  - {checkpoint_base}/{chin}/{run_name}/step0/config.json  (paraphrase only — for tokens_trained)

Writes results/dolma_<size>.py with:
  - data_<S>x = {chinchilla_scale, epochs, flops_multiplier, validation_loss, learning_rate, weight_decay}
  - data_<S>x_para = {chinchilla_scale, K, tokens_trained, flops_multiplier, validation_loss, learning_rate, weight_decay}
  - ALL_DATASETS = [data_<S>x, ...]
  - parap_datasets = [data_<S>x_para, ...]   (omitted if no paraphrase data)

Conventions:
  - N (non-embedding params) is parsed from size label: "30M" -> 30_000_000.
  - 1 chinchilla = 20 * N tokens.
  - Multi-epoch flops_multiplier = chinchilla_scale * epoch.
  - Paraphrase flops_multiplier = tokens_trained / (20 * N) — paraphrase is always 1 epoch over (D + D'*K).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path("/n/home05/sqin/OLMo-core")
RESULTS_ME = REPO / "results" / "dolma_val_loss"
RESULTS_PA = REPO / "results" / "dolma_para_val_loss"
CKPT_BASE = Path("/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints")
OUT_DIR = REPO / "results"

ME_PATTERN = re.compile(
    r"(?P<size>\d+M)_seed42_case4_dolma_epoch(?P<epoch>\d+)_wd(?P<wd>[\d.]+)_lr(?P<lr>[\de.-]+)"
)
PA_PATTERN = re.compile(
    r"(?P<size>\d+M)_seed42_dolma_para_K(?P<K>\d+)_wd(?P<wd>[\d.]+)_lr(?P<lr>[\de.-]+)"
)
SIZE_LABEL = re.compile(r"^(\d+)M$")

CHIN_SCALE_MAP = {
    "chinchilla_0.05": 0.05, "chinchilla_0.1": 0.1, "chinchilla_0.25": 0.25,
    "chinchilla_0.5": 0.5, "chinchilla_1": 1, "chinchilla_2": 2,
    "chinchilla_4": 4, "chinchilla_8": 8, "chinchilla_16": 16,
}
CHIN_ORDER = ["chinchilla_0.05", "chinchilla_0.1", "chinchilla_0.25", "chinchilla_0.5",
              "chinchilla_1", "chinchilla_2", "chinchilla_4", "chinchilla_8", "chinchilla_16"]
VAR_NAMES = {
    "chinchilla_0.05": "data_0_05x", "chinchilla_0.1": "data_0_1x",
    "chinchilla_0.25": "data_0_25x", "chinchilla_0.5": "data_0_5x",
    "chinchilla_1": "data_1x", "chinchilla_2": "data_2x",
    "chinchilla_4": "data_4x", "chinchilla_8": "data_8x", "chinchilla_16": "data_16x",
}


def n_params_from_size(size: str) -> int:
    m = SIZE_LABEL.match(size)
    if not m:
        raise ValueError(f"Bad model size label: {size!r}")
    return int(m.group(1)) * 1_000_000


def load_eval_loss(result_file: Path, run_name: str):
    """Return validation_loss for a result JSON, or None on missing/error."""
    try:
        with open(result_file) as f:
            data = json.load(f)
        if run_name not in data or "error" in data[run_name]:
            return None
        return data[run_name]["validation_loss"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def load_max_duration_tokens(chin_dir: str, run_name: str):
    """Read max_duration (in tokens) from a run's step0/config.json. Returns None on failure."""
    cfg = CKPT_BASE / chin_dir / run_name / "step0" / "config.json"
    try:
        d = json.load(open(cfg))
        md = d["trainer"]["max_duration"]
        if md.get("unit") != "tokens":
            return None
        return int(md["value"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def collect_multi_epoch():
    """Walk RESULTS_ME and return best entry per (size, chin, epoch)."""
    best = {}
    if not RESULTS_ME.exists():
        return best
    for chin_dir in sorted(RESULTS_ME.iterdir()):
        if not chin_dir.is_dir() or chin_dir.name not in CHIN_SCALE_MAP:
            continue
        for size_dir in sorted(chin_dir.iterdir()):
            if not size_dir.is_dir(): continue
            size = size_dir.name
            for result_file in sorted(size_dir.glob("*.json")):
                run_name = result_file.stem
                m = ME_PATTERN.match(run_name)
                if not m: continue
                epoch = int(m.group("epoch")); wd = m.group("wd"); lr = m.group("lr")
                val_loss = load_eval_loss(result_file, run_name)
                if val_loss is None: continue
                key = (size, chin_dir.name, epoch)
                if key not in best or val_loss < best[key]["val_loss"]:
                    best[key] = {"val_loss": val_loss, "lr": lr, "wd": wd, "epoch": epoch}
    return best


def collect_paraphrase():
    """Walk RESULTS_PA, return best entry per (size, chin, K) including tokens_trained."""
    best = {}
    if not RESULTS_PA.exists():
        return best
    for chin_dir in sorted(RESULTS_PA.iterdir()):
        if not chin_dir.is_dir() or chin_dir.name not in CHIN_SCALE_MAP:
            continue
        for size_dir in sorted(chin_dir.iterdir()):
            if not size_dir.is_dir(): continue
            size = size_dir.name
            for result_file in sorted(size_dir.glob("*.json")):
                run_name = result_file.stem
                m = PA_PATTERN.match(run_name)
                if not m: continue
                K = int(m.group("K")); wd = m.group("wd"); lr = m.group("lr")
                val_loss = load_eval_loss(result_file, run_name)
                if val_loss is None: continue
                tokens = load_max_duration_tokens(chin_dir.name, run_name)
                if tokens is None:
                    print(f"  WARN: missing tokens for {chin_dir.name}/{run_name}, skipping")
                    continue
                key = (size, chin_dir.name, K)
                # Tie-break on val_loss; if same K hparam wins, the tokens are identical anyway.
                if key not in best or val_loss < best[key]["val_loss"]:
                    best[key] = {"val_loss": val_loss, "lr": lr, "wd": wd, "K": K,
                                 "tokens_trained": tokens}
    return best


def render_size(size: str, me_best: dict, pa_best: dict) -> str:
    """Render the dolma_<size>.py module content."""
    N = n_params_from_size(size)
    chin_baseline = 20 * N

    # Group by chin
    me_by_chin = defaultdict(list)
    for (s, cd, ep), v in me_best.items():
        if s == size: me_by_chin[cd].append(v)
    pa_by_chin = defaultdict(list)
    for (s, cd, K), v in pa_best.items():
        if s == size: pa_by_chin[cd].append(v)

    for cd in me_by_chin: me_by_chin[cd].sort(key=lambda e: e["epoch"])
    for cd in pa_by_chin: pa_by_chin[cd].sort(key=lambda e: e["K"])

    lines = ["import numpy as np\n"]
    me_vars, pa_vars = [], []

    for cd in CHIN_ORDER:
        if cd in me_by_chin:
            entries = me_by_chin[cd]
            scale = CHIN_SCALE_MAP[cd]
            var = VAR_NAMES[cd]
            me_vars.append(var)
            lines.append(f"{var} = {{")
            lines.append(f"    'chinchilla_scale': {[scale] * len(entries)},")
            lines.append(f"    'epochs': {[e['epoch'] for e in entries]},")
            lines.append(f"    'flops_multiplier': {[scale * e['epoch'] for e in entries]},")
            lines.append(f"    'validation_loss': {[round(e['val_loss'], 4) for e in entries]},")
            lines.append(f"    'learning_rate': [{', '.join(e['lr'] for e in entries)}],")
            lines.append(f"    'weight_decay': [{', '.join(e['wd'] for e in entries)}],")
            lines.append("}\n")

    for cd in CHIN_ORDER:
        if cd in pa_by_chin:
            entries = pa_by_chin[cd]
            scale = CHIN_SCALE_MAP[cd]
            var = f"{VAR_NAMES[cd]}_para"
            pa_vars.append(var)
            tokens_list = [e["tokens_trained"] for e in entries]
            flops_list = [round(t / chin_baseline, 4) for t in tokens_list]
            lines.append(f"{var} = {{")
            lines.append(f"    'chinchilla_scale': {[scale] * len(entries)},")
            lines.append(f"    'K': {[e['K'] for e in entries]},")
            lines.append(f"    'tokens_trained': {tokens_list},")
            lines.append(f"    'flops_multiplier': {flops_list},")
            lines.append(f"    'validation_loss': {[round(e['val_loss'], 4) for e in entries]},")
            lines.append(f"    'learning_rate': [{', '.join(e['lr'] for e in entries)}],")
            lines.append(f"    'weight_decay': [{', '.join(e['wd'] for e in entries)}],")
            lines.append("}\n")

    lines.append("ALL_DATASETS = [")
    for v in me_vars: lines.append(f"    {v},")
    lines.append("]\n")

    if pa_vars:
        lines.append("parap_datasets = [")
        for v in pa_vars: lines.append(f"    {v},")
        lines.append("]")
    else:
        lines.append("parap_datasets = None")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", default=None,
                        help="Restrict to these model sizes (e.g. 30M 60M). Default: all sizes "
                             "with at least one multi-epoch or paraphrase eval.")
    args = parser.parse_args()

    print(f"Walking {RESULTS_ME} ...")
    me_best = collect_multi_epoch()
    print(f"  multi-epoch best entries: {len(me_best)}")
    print(f"Walking {RESULTS_PA} ...")
    pa_best = collect_paraphrase()
    print(f"  paraphrase best entries:  {len(pa_best)}")

    sizes_seen = sorted({k[0] for k in me_best.keys()} | {k[0] for k in pa_best.keys()},
                        key=lambda s: int(s.rstrip("M")))
    sizes = args.sizes if args.sizes else sizes_seen
    if args.sizes:
        unknown = set(args.sizes) - set(sizes_seen)
        if unknown:
            print(f"  WARN: requested sizes not found in eval results: {sorted(unknown)}")

    for size in sizes:
        n_me = sum(1 for k in me_best if k[0] == size)
        n_pa = sum(1 for k in pa_best if k[0] == size)
        if n_me == 0 and n_pa == 0:
            continue
        out = OUT_DIR / f"dolma_{size.lower()}.py"
        out.write_text(render_size(size, me_best, pa_best))
        print(f"Wrote {out}: {n_me} multi-epoch + {n_pa} paraphrase entries")


if __name__ == "__main__":
    main()
