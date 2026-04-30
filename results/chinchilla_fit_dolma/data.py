"""
Shared data-loading / extraction for Dolma pretraining runs.

Each size is a module `dolma_{size}m.py` in ../ exposing an ALL_DATASETS
list of per-scale dicts with keys {chinchilla_scale, epochs, validation_loss}.
"""

import importlib
import os
import sys
from typing import Iterable, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TTP_RATIO = 20  # Chinchilla tokens-per-param ratio (D = scale · TTP · N)

# size → (N in params, importable module name)
SIZES = {
    "14m":  (14e6,  "dolma_14m"),
    "30m":  (30e6,  "dolma_30m"),
    "60m":  (60e6,  "dolma_60m"),
    "100m": (100e6, "dolma_100m"),   # 1-epoch only
    "190m": (190e6, "dolma_190m"),
    "370m": (370e6, "dolma_370m"),
    "600m": (600e6, "dolma_600m"),   # 1-epoch only
}

# u-shape / overfit (scale, epoch) points — exclude when fitting η.
# Rule: loss at this epoch is strictly worse than at the previous epoch.
OVERFIT_EXCLUDE = {
    "14m":  {(0.05, 128), (0.1, 128)},
    "30m":  {(0.05, 128), (0.1, 128)},
    "60m":  {(0.1, 64)},
    "100m": set(),
    "190m": {(0.05, 32), (0.05, 64), (0.1, 32),
             (0.25, 32), (0.5, 32)},
    "370m": {(0.05, 32), (0.05, 64)},
    "600m": set(),
}

# Default scale cut.  We now use 0.0 (i.e. include all scales) and rely on
# residual-based top-k dropping after the first fit, following the
# Besiroglu approach.  Scale_min is kept as a parameter for legacy code paths.
DEFAULT_SCALE_MIN = 0.0


# ──────────────────────────────────────────────────────────────────────

def load(size: str):
    """Return (N, ALL_DATASETS) for a size like '30m', '190m'."""
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; choose from {list(SIZES)}")
    N, module_name = SIZES[size]
    mod = importlib.import_module(module_name)
    return N, mod.ALL_DATASETS


def extract_1epoch(datasets, N: float, scale_min: float = 0.0):
    """Return (scale, D, L) arrays for 1-epoch points at scale >= scale_min."""
    rows = []
    for ds in datasets:
        scale = ds["chinchilla_scale"][0]
        if scale < scale_min:
            continue
        if 1 not in ds["epochs"]:
            continue
        idx = ds["epochs"].index(1)
        loss = ds["validation_loss"][idx]
        if np.isnan(loss):
            continue
        rows.append((scale, scale * TTP_RATIO * N, loss))
    if not rows:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2]


def extract_multi_epoch(datasets, N: float, *,
                        scale_min: float = DEFAULT_SCALE_MIN,
                        exclude_overfit: Optional[Set[Tuple[float, int]]] = None):
    """Return (scale, D, epochs, D', L) for multi-epoch (e>1) points.

    exclude_overfit: set of (scale, epoch) pairs to drop (default: empty).
    """
    exclude_overfit = exclude_overfit or set()
    rows = []
    for ds in datasets:
        scale = ds["chinchilla_scale"][0]
        if scale < scale_min:
            continue
        D = scale * TTP_RATIO * N
        for i, ep in enumerate(ds["epochs"]):
            if ep == 1:
                continue
            if (scale, ep) in exclude_overfit:
                continue
            loss = ds["validation_loss"][i]
            if np.isnan(loss):
                continue
            rows.append((scale, D, ep, (ep - 1) * D, loss))
    if not rows:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def all_sizes() -> Iterable[str]:
    return list(SIZES.keys())
