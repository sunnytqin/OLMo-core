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
    """Return (scale, D, epochs, D', L, L_1ep) for multi-epoch (e>1) points.

    L_1ep is the observed 1-epoch loss at the same (size, scale) — used by
    the loss-difference η solver (which avoids E_eff dependence).

    exclude_overfit: set of (scale, epoch) pairs to drop (default: empty).
    Multi-epoch rows are skipped if no 1-epoch loss exists at that scale.
    """
    exclude_overfit = exclude_overfit or set()
    rows = []
    for ds in datasets:
        scale = ds["chinchilla_scale"][0]
        if scale < scale_min:
            continue
        if 1 not in ds["epochs"]:
            continue
        L_1ep = ds["validation_loss"][ds["epochs"].index(1)]
        if np.isnan(L_1ep):
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
            rows.append((scale, D, ep, (ep - 1) * D, loss, L_1ep))
    if not rows:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


def all_sizes() -> Iterable[str]:
    return list(SIZES.keys())


# ──────────────────────────────────────────────────────────────────────
# Paraphrase data (parallel to multi-epoch).
#
# A paraphrase run trains on D fresh Dolma tokens + D'_para paraphrased
# tokens (≈ K · α · D, with α ≈ 0.3 because paraphrases are shorter than
# originals; we read D'_para directly off `tokens_trained − D`).
#
# Same η fitting framework: L = E_eff(N) + B/(D + η_para · D')^β, with
# the joint Chinchilla anchors fixed.  L_1ep is the *no-paraphrase*
# 1-epoch loss at the same (size, scale), used by the ΔL solver.
# ──────────────────────────────────────────────────────────────────────

def extract_paraphrase(datasets, parap_datasets, N: float, *,
                        scale_min: float = DEFAULT_SCALE_MIN):
    """Return (scale, D, K, D'_para, L, L_1ep) arrays for paraphrase points.

    Skips rows where no 1-epoch (no-paraphrase) baseline exists at the
    same scale, so the ΔL solver and the loss-difference diagnostics
    have a 1-epoch reference to compare against.
    """
    L_1ep_by_scale = {}
    for ds in datasets:
        if 1 not in ds["epochs"]:
            continue
        scale = ds["chinchilla_scale"][0]
        L_1ep_by_scale[scale] = ds["validation_loss"][ds["epochs"].index(1)]

    rows = []
    for pds in parap_datasets:
        scale = pds["chinchilla_scale"][0]
        if scale < scale_min:
            continue
        if scale not in L_1ep_by_scale:
            continue
        L_1ep = L_1ep_by_scale[scale]
        if np.isnan(L_1ep):
            continue
        D = scale * TTP_RATIO * N
        for i, K in enumerate(pds["K"]):
            tokens_trained = pds["tokens_trained"][i]
            loss = pds["validation_loss"][i]
            if np.isnan(loss):
                continue
            Dp = float(tokens_trained) - D
            if Dp <= 0:
                continue
            rows.append((scale, D, int(K), Dp, loss, L_1ep))
    if not rows:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


def load_with_para(size: str):
    """Return (N, ALL_DATASETS, parap_datasets) for a size.

    Sizes without a `parap_datasets` attribute return an empty list.
    """
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; choose from {list(SIZES)}")
    N, module_name = SIZES[size]
    mod = importlib.import_module(module_name)
    return N, mod.ALL_DATASETS, getattr(mod, "parap_datasets", [])


def load_with_extras(size: str):
    """Return (N, ALL_DATASETS, parap_datasets, selfdistill_datasets)."""
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; choose from {list(SIZES)}")
    N, module_name = SIZES[size]
    mod = importlib.import_module(module_name)
    return (N, mod.ALL_DATASETS,
            getattr(mod, "parap_datasets", []),
            getattr(mod, "selfdistill_datasets", []))


def extract_selfdistill(datasets, sd_datasets, N: float, *,
                         scale_min: float = DEFAULT_SCALE_MIN):
    """Return (scale, D, K, D'_sd, L, L_1ep) for self-distill rows.

    Same convention as `extract_paraphrase`: D = chinchilla_scale · 20 · N
    fresh tokens; D'_sd = tokens_trained − D self-distilled tokens added on
    top.  For 30M self-distill, tokens_trained = 2K · D so D'/D = 2K − 1
    (range 1 → 31 for K ∈ {1, 2, 4, 8, 16}).
    """
    L_1ep_by_scale = {}
    for ds in datasets:
        if 1 not in ds["epochs"]:
            continue
        scale = ds["chinchilla_scale"][0]
        L_1ep_by_scale[scale] = ds["validation_loss"][ds["epochs"].index(1)]

    rows = []
    for sds in sd_datasets:
        scale = sds["chinchilla_scale"][0]
        if scale < scale_min:
            continue
        if scale not in L_1ep_by_scale:
            continue
        L_1ep = L_1ep_by_scale[scale]
        if np.isnan(L_1ep):
            continue
        D = scale * TTP_RATIO * N
        for i, K in enumerate(sds["K"]):
            tokens_trained = sds["tokens_trained"][i]
            loss = sds["validation_loss"][i]
            if np.isnan(loss):
                continue
            Dp = float(tokens_trained) - D
            if Dp <= 0:
                continue
            rows.append((scale, D, int(K), Dp, loss, L_1ep))
    if not rows:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
