#!/usr/bin/env python3
"""
Aggregate dolma_val_loss results and regenerate results/dolma_<size>.py
with the best (lowest loss) hparam run for each (chinchilla_scale, epoch).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_BASE = Path("/n/home05/sqin/OLMo-core/results/dolma_val_loss")
OUT_DIR = Path("/n/home05/sqin/OLMo-core/results")

CHIN_VALUES = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
MODEL_SIZES = ["30M", "60M", "190M", "370M"]

RUN_RE = re.compile(
    r"(?P<size>\d+M)_seed42_case4_dolma_epoch(?P<epoch>\d+)_wd(?P<wd>[\d.]+)_lr(?P<lr>[\de.-]+)"
)


def load_best() -> dict:
    """Return best[size][chin][epoch] = (loss, lr, wd)."""
    best = defaultdict(lambda: defaultdict(dict))
    for chin in CHIN_VALUES:
        chin_str = f"chinchilla_{chin:g}"
        chin_dir = RESULTS_BASE / chin_str
        if not chin_dir.is_dir():
            continue
        for size in MODEL_SIZES:
            size_dir = chin_dir / size
            if not size_dir.is_dir():
                continue
            for f in size_dir.glob("*.json"):
                m = RUN_RE.match(f.stem)
                if not m:
                    continue
                try:
                    data = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                entry = data.get(f.stem)
                if not entry or "error" in entry:
                    continue
                loss = entry.get("validation_loss")
                if loss is None:
                    continue
                epoch = int(m.group("epoch"))
                lr = m.group("lr")
                wd = m.group("wd")
                cur = best[size][chin].get(epoch)
                if cur is None or loss < cur[0]:
                    best[size][chin][epoch] = (loss, lr, wd)
    return best


def fmt_lr(s: str) -> str:
    return s


def fmt_wd(s: str) -> str:
    return s


def render_dataset(chin: float, ep_map: dict) -> str:
    epochs = sorted(ep_map.keys())
    chin_list = [chin] * len(epochs)
    flops = [chin * e for e in epochs]
    losses = [round(ep_map[e][0], 4) for e in epochs]
    lrs = [ep_map[e][1] for e in epochs]
    wds = [ep_map[e][2] for e in epochs]

    def lst(xs, quote=False):
        if quote:
            return "[" + ", ".join(f"'{x}'" for x in xs) + "]"
        return "[" + ", ".join(str(x) for x in xs) + "]"

    return (
        f"    'chinchilla_scale': {lst(chin_list)},\n"
        f"    'epochs': {lst(epochs)},\n"
        f"    'flops_multiplier': {lst(flops)},\n"
        f"    'validation_loss': {lst(losses)},\n"
        f"    'learning_rate': [" + ", ".join(lrs) + "],\n"
        f"    'weight_decay': [" + ", ".join(wds) + "],\n"
    )


def render_file(size: str, chin_map: dict) -> str:
    chins_present = sorted(c for c in chin_map if chin_map[c])
    var_names = []
    body = ["import numpy as np\n"]
    for chin in chins_present:
        # Variable name: data_0_05x, data_1x, data_16x, etc.
        if chin < 1:
            # e.g., 0.05 -> 0_05, 0.1 -> 0_1, 0.25 -> 0_25
            s = f"{chin:g}".replace(".", "_")
            var = f"data_{s}x"
        else:
            var = f"data_{int(chin)}x"
        var_names.append(var)
        body.append(f"{var} = {{\n{render_dataset(chin, chin_map[chin])}}}\n")

    body.append("ALL_DATASETS = [")
    for v in var_names:
        body.append(f"    {v},")
    body.append("]\n")
    return "\n".join(body)


def main():
    best = load_best()
    for size in MODEL_SIZES:
        if size not in best:
            print(f"no data for {size}, skipping")
            continue
        content = render_file(size, best[size])
        out = OUT_DIR / f"dolma_{size.lower()}.py"
        out.write_text(content)

        # summary
        total = sum(len(v) for v in best[size].values())
        chins = sorted(best[size].keys())
        print(f"{size}: {total} (chin, epoch) entries across {len(chins)} chinchilla scales -> {out.name}")
        for c in chins:
            eps = sorted(best[size][c].keys())
            print(f"  {c}x: epochs {eps}")


if __name__ == "__main__":
    main()
