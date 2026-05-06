"""
Variation of §6's triple-joint fit: drop all *non-paraphrase* 14M points
from the training set (i.e. drop 14M 1-epoch and 14M repetition rows),
but keep all 14M paraphrase rows + everything for every other size.

The hypothesis being tested: in the §6 fit, 14M's high-residual 1-ep /
rep points may be holding the paraphrase σ exponent at the
counter-intuitive +0.18.  Without 14M non-para points, the Chinchilla
curve is pinned by N ≥ 30M data; the 14M paraphrase rows act purely as
a small-N anchor for the paraphrase η surface.  Does σ_para flip
negative once the 14M Chinchilla bias is removed?

Output:
  fit_joint_triple_drop14m.pdf — three-panel diagnostic.
  Side-by-side parameter / RMSE comparison vs the unmodified §6 fit.
"""

import glob
import os
import sys

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

from data import SIZES  # noqa: E402
from fit_joint_triple import (DELTA, REF_ONESHOT_REP, SOURCE_NONE,  # noqa: E402
                                SOURCE_REPEAT, SOURCE_PARA, _print_params,
                                _summary, collect_pooled_triple,
                                make_triple_forward,
                                plot_triple_diagnostic, stage1_rep_only,
                                stage2_triple, topk_drop_sweep)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reproduce §6 numbers as a comparison anchor (see writeup §6.2)
REF_TRIPLE_K15 = dict(
    E=0.003, A=28.9, B=15599, alpha=0.133, beta=0.431,
    log_K_rep=10.58, rho_rep=-0.414, sigma_rep=-0.394,
    log_K_para=10.10, rho_para=-2.555, sigma_para=+0.177,
)


def filter_data(data, mask):
    return {
        "tags":   data["tags"][mask],
        "N":      data["N"][mask],
        "D":      data["D"][mask],
        "Dp":     data["Dp"][mask],
        "L":      data["L"][mask],
        "source": data["source"][mask],
    }


def main():
    print("=" * 96)
    print("Triple-joint fit, dropping non-paraphrase 14M points "
          "(14M 1-ep + 14M rep) from the training set")
    print("=" * 96)

    full = collect_pooled_triple(scale_min_para=0.5)
    drop_mask = (full["tags"] == "14m") & (full["source"] != SOURCE_PARA)
    keep_mask = ~drop_mask
    n_dropped = int(drop_mask.sum())
    n_kept = int(keep_mask.sum())
    print(f"\nFull pool n={len(full['L'])}  "
          f"(1ep={int((full['source']==SOURCE_NONE).sum())}, "
          f"rep={int((full['source']==SOURCE_REPEAT).sum())}, "
          f"para={int((full['source']==SOURCE_PARA).sum())})")
    print(f"Dropping 14M non-para: {n_dropped} points  "
          f"(14M 1ep + 14M rep)")
    print(f"Fit set: n={n_kept}")
    for tag in sorted(set(full["tags"]), key=lambda t: SIZES[t][0]):
        m = (full["tags"] == tag) & keep_mask
        if not m.any(): continue
        n1 = int((m & (full["source"] == SOURCE_NONE)).sum())
        nr = int((m & (full["source"] == SOURCE_REPEAT)).sum())
        np_ = int((m & (full["source"] == SOURCE_PARA)).sum())
        print(f"  {tag}: 1ep={n1}, rep={nr}, para={np_}  (total {int(m.sum())})")

    fit_data = filter_data(full, keep_mask)

    # ── Same §6 pipeline: stage 1 (rep+1ep only on this filtered set)
    #    → stage 2 (warm-start + para grid) → residual drop
    print("\n[Stage 1] rep-only sub-model (9 params)...")
    res1 = stage1_rep_only(fit_data); p1 = res1["params"]
    _print_params("(stage 1)", p1)

    print("\n[Stage 2] triple model (warm-start + para grid)...")
    res2 = stage2_triple(fit_data, p1); p2 = res2["params"]
    _print_params("(stage 2)", p2)

    K_VALUES = [0, 5, 10, 15, 20, 25, 30]
    print(f"\n[Stage 3] iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(fit_data, p2, K_VALUES)

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

    # Canonical k=15 (matching §6 / writeup_final convention)
    canonical = 15
    rc = sweep[canonical]
    sc = _summary(fit_data, rc["params"], rc["pred_full"], rc["keep"])
    print(f"\n  Canonical k={canonical} (matching §6 / writeup_final):")
    _print_params(f"(drop-14m-non-para, k={canonical})", rc["params"])
    print(f"    RMSE — 1ep:{sc['rmse_1ep']:.4f}  rep:{sc['rmse_rep']:.4f}  "
          f"para:{sc['rmse_para']:.4f}  |  kept:{sc['rmse_kept']:.4f}")

    # ── Comparison table vs §6 unmodified ──────────────────────────
    print("\n" + "=" * 96)
    print("Side-by-side: §6 unmodified vs this variation (both k=15)")
    print("=" * 96)
    print(f"  {'param':<14s}  {'§6 (all data)':>14s}  "
          f"{'drop 14M non-para':>20s}  {'Δ':>8s}")
    pc = rc["params"]
    rows = [
        ("E",            REF_TRIPLE_K15["E"],          float(np.exp(pc["e"]))),
        ("A",            REF_TRIPLE_K15["A"],          float(np.exp(pc["a"]))),
        ("B",            REF_TRIPLE_K15["B"],          float(np.exp(pc["b"]))),
        ("alpha",        REF_TRIPLE_K15["alpha"],      pc["alpha"]),
        ("beta",         REF_TRIPLE_K15["beta"],       pc["beta"]),
        ("log_K_rep",    REF_TRIPLE_K15["log_K_rep"],  pc["log_K_rep"]),
        ("rho_rep",      REF_TRIPLE_K15["rho_rep"],    pc["rho_rep"]),
        ("sigma_rep",    REF_TRIPLE_K15["sigma_rep"],  pc["sigma_rep"]),
        ("log_K_para",   REF_TRIPLE_K15["log_K_para"], pc["log_K_para"]),
        ("rho_para",     REF_TRIPLE_K15["rho_para"],   pc["rho_para"]),
        ("sigma_para",   REF_TRIPLE_K15["sigma_para"], pc["sigma_para"]),
    ]
    for name, ref, new in rows:
        delta = new - ref
        # Special formatting for big-number rows
        if name in ("E", "A", "B"):
            print(f"  {name:<14s}  {ref:>14.4g}  {new:>20.4g}  "
                  f"{delta:>+8.4g}")
        else:
            print(f"  {name:<14s}  {ref:>+14.4f}  {new:>+20.4f}  "
                  f"{delta:>+8.4f}")

    plot_triple_diagnostic(
        fit_data, rc["pred_full"], rc["keep"], rc["dropped"],
        rc["params"], sc,
        path=os.path.join(SCRIPT_DIR, "fit_joint_triple_drop14m.pdf"))

    return sweep, canonical


if __name__ == "__main__":
    main()
