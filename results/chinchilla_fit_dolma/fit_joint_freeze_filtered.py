"""Frozen-Chinchilla η_para fit with per-point η > 1 filter.

Same as fit_joint_freeze.py, but before the η_para fit we drop paraphrase
points whose per-point ΔL solver returns η > 1.  Those points are
inconsistent with the physical bound η ≤ 1 (paraphrase tokens cannot be
more valuable than fresh tokens) and almost all live in a TTP ≈ 16–64
band where small Chinchilla-fit residuals on L_1 get levered into
apparent η > 1 by the ΔL inversion.

Functional form (paraphrase points only):

    L(N, D, D'; para) = E + A/N^α + B / (D + η_para(D, D'; N) · D')^β
    η_para(D, D'; N) = R*_para · (1 − e^{−x/R*_para}) / x,   x = D'/D
    log R*_para     = log K_para + ρ_para · log(D/N) + σ_para · log N

Frozen (writeup_final, k=15):
    E = 0.050,  A = 31,  B = 16539,  α = 0.137,  β = 0.436

Pipeline:
    1. Load paraphrase data.
    2. Compute per-point η via ΔL solver (frozen B, β).
    3. Drop points with η_pp > 1, NaN, or solver failure (denom ≤ 0).
    4. Stage 2: warm-start fit (log_K_para, ρ_para, σ_para).
    5. Stage 3: iterative residual-greedy drop sweep on the filtered pool.

Headline (canonical k=5): log K = 8.87, ρ = -0.14, σ = -0.40,
para RMSE 0.042.

Output:
    fit_joint_freeze_filtered.json — canonical-k summary.
    fit_joint_freeze_filtered.pdf  — diagnostic plot.
"""

import glob
import json
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))

# Palatino styling
for _f in glob.glob("/usr/share/fonts/urw-base35/P052-*.otf"):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["P052", "Palatino", "TeX Gyre Pagella", "serif"],
    "mathtext.fontset": "cm",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.titlesize": 14,
})

from data import SIZES, extract_paraphrase, load_with_para  # noqa: E402
from fit_joint_freeze import (DELTA, EXCLUDE_SIZES, FROZEN_FULL,  # noqa: E402
                                FROZEN_A, FROZEN_ALPHA, FROZEN_B,
                                FROZEN_BETA, FROZEN_E, FROZEN_LOG_K_REP,
                                FROZEN_RHO_REP, FROZEN_SIGMA_REP,
                                _print_params, _summary, _wrap_fixed,
                                plot_diagnostic, stage2_para_only,
                                topk_drop_sweep)
from fit_joint_triple import SOURCE_PARA, fmt_tokens  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def pick_canonical_k_param_plateau(sweep, k_values, tol_logK=0.1,
                                     tol_sigma=0.05):
    """Largest k still on the (log_K, σ) plateau seeded at k_values[0].

    log_K and σ are the structurally informative anchors (intercept and
    N-scaling of R*); ρ is correlated with log_K and shifts smoothly under
    residual drop, so we don't gate on it.  Walks forward; the canonical
    is the last k before either log_K or σ moves more than the tolerance.
    """
    last_on_plateau = k_values[0]
    for i in range(1, len(k_values)):
        p_prev = sweep[k_values[i - 1]]["params"]
        p_curr = sweep[k_values[i]]["params"]
        moved = (
            abs(p_curr["log_K_para"] - p_prev["log_K_para"]) > tol_logK
            or abs(p_curr["sigma_para"] - p_prev["sigma_para"]) > tol_sigma
        )
        if moved:
            break
        last_on_plateau = k_values[i]
    return last_on_plateau


# ──────────────────────────────────────────────────────────────────────
# Data: paraphrase, with per-point η > 1 filter
# ──────────────────────────────────────────────────────────────────────

def per_point_eta(D: float, Dp: float, L: float, L1: float,
                  B: float, beta: float) -> float:
    """Solve for the per-point η implied by ΔL = L1 - L under
    L = E_eff(N) + B/(D + η D')^β.  Returns NaN if the solver fails."""
    dL = L1 - L
    denom = (1.0 / D ** beta) - (dL / B)
    if denom <= 0:
        return float("nan")
    D_eff = denom ** (-1.0 / beta)
    return (D_eff - D) / Dp


def collect_para_filtered(eta_max: float = 1.0,
                            B: float = FROZEN_B,
                            beta: float = FROZEN_BETA):
    """Pool paraphrase points across sizes (excluding EXCLUDE_SIZES) and
    drop any whose per-point η solver returns NaN or η > eta_max.

    Returns (data_dict, drop_log) where drop_log is a list of
    (size, scale, D, Dp, L, L1, eta_pp) for every dropped row.
    """
    tags, Ns, Ds, Dps, Ls, src = [], [], [], [], [], []
    drop_log = []
    keep_count = 0
    total_count = 0
    for size in SIZES:
        if size in EXCLUDE_SIZES:
            continue
        N, datasets, parap = load_with_para(size)
        if not parap:
            continue
        s_, D_, _K, Dp_, L_, L1_ = extract_paraphrase(
            datasets, parap, N, scale_min=0.0)
        for s, d, dp, l, l1 in zip(s_, D_, Dp_, L_, L1_):
            total_count += 1
            eta_pp = per_point_eta(d, dp, l, l1, B, beta)
            if np.isnan(eta_pp) or eta_pp > eta_max:
                drop_log.append(dict(size=size, N=N, scale=float(s),
                                       D=float(d), Dp=float(dp), L=float(l),
                                       L1=float(l1), eta_pp=float(eta_pp)))
                continue
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
            Ls.append(l); src.append(SOURCE_PARA)
            keep_count += 1
    data = dict(
        tags=np.array(tags),
        N=np.array(Ns, dtype=np.float64),
        D=np.array(Ds, dtype=np.float64),
        Dp=np.array(Dps, dtype=np.float64),
        L=np.array(Ls, dtype=np.float64),
        source=np.array(src, dtype=np.int64),
    )
    return data, drop_log, total_count


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def print_drop_log(drop_log, total_count):
    print(f"\n[Filter] Dropped {len(drop_log)} / {total_count} points "
          f"with per-point η > 1 (or solver failure).")
    by_size: Dict[str, int] = {}
    for r in drop_log:
        by_size[r["size"]] = by_size.get(r["size"], 0) + 1
    if by_size:
        print("  Drops per size: " +
              ", ".join(f"{k}: {v}" for k, v in sorted(
                  by_size.items(), key=lambda kv: SIZES[kv[0]][0])))
    print(f"\n  {'size':>5}  {'scale':>5}  {'TTP':>5}  "
          f"{f'D{chr(39)}/D':>5}  {'L1':>6}  {'L_para':>6}  {'eta_pp':>7}")
    for r in sorted(drop_log,
                     key=lambda x: (SIZES[x["size"]][0], x["scale"])):
        print(f"  {r['size']:>5}  {r['scale']:>5.2f}  "
              f"{r['D']/r['N']:>5.0f}  {r['Dp']/r['D']:>5.1f}  "
              f"{r['L1']:>6.3f}  {r['L']:>6.3f}  {r['eta_pp']:>+7.3f}")


def main():
    print("=" * 96)
    print("Frozen-Chinchilla η_para fit  (per-point η > 1 filter applied)")
    print("=" * 96)

    print("\n[Stage 1] FROZEN — writeup_final headline values:")
    print(f"  E={FROZEN_E}, A={FROZEN_A}, B={FROZEN_B}, "
          f"α={FROZEN_ALPHA}, β={FROZEN_BETA}")
    print(f"  η_rep (frozen, unused): logK={FROZEN_LOG_K_REP}, "
          f"ρ={FROZEN_RHO_REP}, σ={FROZEN_SIGMA_REP}")

    data, drop_log, total = collect_para_filtered(
        eta_max=1.0, B=FROZEN_B, beta=FROZEN_BETA)
    print_drop_log(drop_log, total)

    excl = sorted(EXCLUDE_SIZES) if EXCLUDE_SIZES else "(none)"
    print(f"\n[Pool] After filter: n={len(data['L'])} / {total}  "
          f"[excluded sizes: {excl}]")
    sizes_seen = sorted(set(data["tags"].tolist()),
                        key=lambda t: SIZES[t][0])
    counts = {t: int((data["tags"] == t).sum()) for t in sizes_seen}
    print(f"  per-size counts (kept): {counts}")

    fixed_params = dict(FROZEN_FULL)

    print(f"\n[Stage 2] Fitting η_para only "
          f"(8 frozen params held fixed, 3 trainable)...")
    res2 = stage2_para_only(data, fixed_params)
    para_init = res2["params"]
    p2 = {**para_init, **fixed_params}
    _print_params("(stage 2 — η_para over frozen Stage 1, filtered pool)", p2)

    K_VALUES = [0, 2, 5, 8, 10, 12, 15]
    print(f"\n[Stage 3] Iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(data, para_init, fixed_params, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  | "
          f"{'lK_par':>7s} {'rho_par':>7s} {'sig_par':>7s} | "
          f"{'para':>5s}")
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(data, r["params"], r["pred_full"], r["keep"])
        p = r["params"]
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  | "
              f"{p['log_K_para']:>7.2f} {p['rho_para']:>+7.3f} "
              f"{p['sigma_para']:>+7.3f} | "
              f"{s['rmse_para']:>5.3f}")

    canonical = pick_canonical_k_param_plateau(sweep, K_VALUES)
    print(f"\n  Canonical k (parameter plateau): k = {canonical}")
    rc = sweep[canonical]
    sc = _summary(data, rc["params"], rc["pred_full"], rc["keep"])
    _print_params(f"(canonical k={canonical})", rc["params"])
    print(f"    para RMSE (kept): {sc['rmse_para']:.4f}  |  "
          f"all-points RMSE: {sc['rmse_full']:.4f}")

    out_json = {
        "method": "frozen-chinchilla, paraphrase-only η_para fit, "
                   "η>1 violator filter",
        "filter": {
            "type": "per-point η > 1 (ΔL solver, frozen B+β)",
            "B_frozen": FROZEN_B,
            "beta_frozen": FROZEN_BETA,
            "n_total": total,
            "n_dropped": len(drop_log),
            "n_kept": int(len(data["L"])),
            "drops": drop_log,
        },
        "excluded_sizes": sorted(EXCLUDE_SIZES),
        "frozen": dict(FROZEN_FULL),
        "frozen_human": {
            "E": FROZEN_E, "A": FROZEN_A, "B": FROZEN_B,
            "alpha": FROZEN_ALPHA, "beta": FROZEN_BETA,
            "log_K_rep": FROZEN_LOG_K_REP,
            "rho_rep": FROZEN_RHO_REP,
            "sigma_rep": FROZEN_SIGMA_REP,
        },
        "k_values": K_VALUES,
        "canonical_k": canonical,
        "canonical_params": rc["params"],
        "canonical_summary": sc,
        "per_k": {
            str(k): {
                "params": sweep[k]["params"],
                "summary": _summary(data, sweep[k]["params"],
                                     sweep[k]["pred_full"],
                                     sweep[k]["keep"]),
                "n_dropped": len(sweep[k]["dropped"]),
            } for k in K_VALUES
        },
        "n_total_after_filter": int(len(data["L"])),
        "per_size_counts": counts,
    }
    json_path = os.path.join(SCRIPT_DIR, "fit_joint_freeze_filtered.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved {json_path}")

    plot_diagnostic(
        data, rc["pred_full"], rc["keep"], rc["dropped"],
        rc["params"], sc,
        path=os.path.join(SCRIPT_DIR, "fit_joint_freeze_filtered.pdf"))

    return sweep, canonical


if __name__ == "__main__":
    main()
