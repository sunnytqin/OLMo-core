"""
Same as fit_joint_triple_drop14m.py (drop 14M non-paraphrase points
from training set), but with a *single fresh grid search over all 11
parameters* — no staged warm-start.  Critically, the σ_para grid
brackets zero so the optimizer can land on either sign.

If σ_para flips negative without the staged warm-start, then §6's
+0.16 was a basin-of-attraction artefact.  If σ_para stays positive,
the data really wants it positive.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(__file__))

# Palatino styling
for _f in glob.glob("/usr/share/fonts/urw-base35/P052-*.otf"):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["P052", "Palatino", "TeX Gyre Pagella", "serif"],
    "mathtext.fontset": "cm",
    "font.size":        12,
})

from data import SIZES  # noqa: E402
from fit_joint_triple import (DELTA, SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA,  # noqa: E402
                                _print_params, _summary,
                                collect_pooled_triple, make_triple_forward,
                                plot_triple_diagnostic, topk_drop_sweep)
from fit_lse import expand_grid, fit_lse  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reproduce §6 numbers as a comparison anchor
REF_TRIPLE_K15 = dict(
    E=0.003, A=28.9, B=15599, alpha=0.133, beta=0.431,
    log_K_rep=10.58, rho_rep=-0.414, sigma_rep=-0.394,
    log_K_para=10.10, rho_para=-2.555, sigma_para=+0.177,
)


def filter_data(data, mask):
    return {k: v[mask] for k, v in data.items()}


def fresh_full_grid_fit(data, delta=DELTA):
    """Single fresh grid search across all 11 params.

    σ_para grid is symmetric around 0 ([-1, 0, +1]) so the optimizer
    can pick either sign without bias.  Same for ρ_para and σ_rep.
    Total inits: 2^8 · 3 · 3 = 2304.
    """
    grid = expand_grid({
        "e":            [0.0, 1.0],
        "a":            [3.0, 12.0],
        "b":            [5.0, 12.0],
        "alpha":        [0.15, 0.4],
        "beta":         [0.3, 0.5],
        "log_K_rep":    [10.0, 18.0],
        "rho_rep":      [-1.0, 0.0],
        "sigma_rep":    [-1.0, 0.0],
        "log_K_para":   [10.0, 18.0],
        "rho_para":     [-2.0, 0.0, +1.0],
        "sigma_para":   [-1.0, 0.0, +1.0],
    })
    print(f"  fresh grid size: {len(grid)} inits")
    fwd = make_triple_forward(data["N"], data["D"], data["Dp"], data["source"])
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    return fit_lse(fwd, log_L, grid, delta=delta, verbose=False)


def main():
    print("=" * 96)
    print("Fresh-grid triple fit (no warm-start), 14M non-para dropped")
    print("=" * 96)

    full = collect_pooled_triple(scale_min_para=0.5)
    drop_mask = (full["tags"] == "14m") & (full["source"] != SOURCE_PARA)
    keep_mask = ~drop_mask
    fit_data = filter_data(full, keep_mask)
    print(f"\nFit set: n={len(fit_data['L'])}  "
          f"(dropped {int(drop_mask.sum())} 14M non-para points)")

    print("\n[Fresh grid] single-stage 11-param fit on the filtered data...")
    res = fresh_full_grid_fit(fit_data)
    p_fresh = res["params"]
    _print_params("(fresh-grid, k=0)", p_fresh)

    K_VALUES = [0, 5, 10, 15, 20, 25, 30]
    print(f"\n[Residual drop] iterative greedy, k ∈ {K_VALUES}, "
          f"warm-started from fresh-grid optimum...")
    sweep = topk_drop_sweep(fit_data, p_fresh, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  {'E':>7s}  {'A':>7s}  "
          f"{'B':>9s}  {'α':>6s}  {'β':>6s}  | "
          f"{'lK_r':>6s} {'ρ_r':>6s} {'σ_r':>6s} | "
          f"{'lK_p':>6s} {'ρ_p':>6s} {'σ_p':>6s} | "
          f"{'1ep':>5s}  {'rep':>5s}  {'par':>5s}")
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(fit_data, r["params"], r["pred_full"], r["keep"])
        p = r["params"]
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  "
              f"{np.exp(p['e']):>7.3f}  {np.exp(p['a']):>7.1f}  "
              f"{np.exp(p['b']):>9.0f}  {p['alpha']:>6.3f}  "
              f"{p['beta']:>6.3f}  | "
              f"{p['log_K_rep']:>6.2f} {p['rho_rep']:>+6.2f} "
              f"{p['sigma_rep']:>+6.2f} | "
              f"{p['log_K_para']:>6.2f} {p['rho_para']:>+6.2f} "
              f"{p['sigma_para']:>+6.2f} | "
              f"{s['rmse_1ep']:>5.3f}  {s['rmse_rep']:>5.3f}  "
              f"{s['rmse_para']:>5.3f}")

    canonical = 15
    rc = sweep[canonical]
    sc = _summary(fit_data, rc["params"], rc["pred_full"], rc["keep"])
    print(f"\n  Canonical k={canonical}:")
    _print_params(f"(fresh-grid, k={canonical})", rc["params"])
    print(f"    RMSE — 1ep:{sc['rmse_1ep']:.4f}  rep:{sc['rmse_rep']:.4f}  "
          f"para:{sc['rmse_para']:.4f}  |  kept:{sc['rmse_kept']:.4f}")

    # ── Side-by-side comparison ────────────────────────────────────
    # Read the warm-started variant's k=15 params from the analogous run.
    # (Hard-coded from the previous run — same data, just a different
    # init pipeline.)
    REF_DROP14M_WARM_K15 = dict(
        E=1.5099, A=326.66, B=36766.40, alpha=0.3082, beta=0.4778,
        log_K_rep=10.59, rho_rep=-0.195, sigma_rep=-0.414,
        log_K_para=10.11, rho_para=-2.247, sigma_para=+0.160,
    )

    print("\n" + "=" * 96)
    print("Comparison @ k=15:  §6 (all data)  vs  drop-14M-non-para (warm)  "
          "vs  drop-14M-non-para (fresh)")
    print("=" * 96)
    pc = rc["params"]
    rows = [
        ("E",          REF_TRIPLE_K15["E"],          REF_DROP14M_WARM_K15["E"],          float(np.exp(pc["e"]))),
        ("A",          REF_TRIPLE_K15["A"],          REF_DROP14M_WARM_K15["A"],          float(np.exp(pc["a"]))),
        ("B",          REF_TRIPLE_K15["B"],          REF_DROP14M_WARM_K15["B"],          float(np.exp(pc["b"]))),
        ("alpha",      REF_TRIPLE_K15["alpha"],      REF_DROP14M_WARM_K15["alpha"],      pc["alpha"]),
        ("beta",       REF_TRIPLE_K15["beta"],       REF_DROP14M_WARM_K15["beta"],       pc["beta"]),
        ("log_K_rep",  REF_TRIPLE_K15["log_K_rep"],  REF_DROP14M_WARM_K15["log_K_rep"],  pc["log_K_rep"]),
        ("rho_rep",    REF_TRIPLE_K15["rho_rep"],    REF_DROP14M_WARM_K15["rho_rep"],    pc["rho_rep"]),
        ("sigma_rep",  REF_TRIPLE_K15["sigma_rep"],  REF_DROP14M_WARM_K15["sigma_rep"],  pc["sigma_rep"]),
        ("log_K_para", REF_TRIPLE_K15["log_K_para"], REF_DROP14M_WARM_K15["log_K_para"], pc["log_K_para"]),
        ("rho_para",   REF_TRIPLE_K15["rho_para"],   REF_DROP14M_WARM_K15["rho_para"],   pc["rho_para"]),
        ("sigma_para", REF_TRIPLE_K15["sigma_para"], REF_DROP14M_WARM_K15["sigma_para"], pc["sigma_para"]),
    ]
    print(f"  {'param':<12s}  {'§6':>10s}  {'drop14m warm':>14s}  "
          f"{'drop14m fresh':>15s}")
    for name, ref_a, ref_b, val in rows:
        if name in ("E", "A", "B"):
            print(f"  {name:<12s}  {ref_a:>10.4g}  {ref_b:>14.4g}  "
                  f"{val:>15.4g}")
        else:
            print(f"  {name:<12s}  {ref_a:>+10.4f}  {ref_b:>+14.4f}  "
                  f"{val:>+15.4f}")

    plot_triple_diagnostic(
        fit_data, rc["pred_full"], rc["keep"], rc["dropped"],
        rc["params"], sc,
        path=os.path.join(SCRIPT_DIR, "fit_joint_triple_drop14m_freshgrid.pdf"))

    return sweep, canonical


if __name__ == "__main__":
    main()
