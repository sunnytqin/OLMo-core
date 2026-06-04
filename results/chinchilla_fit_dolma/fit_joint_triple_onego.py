"""
One-go joint triple fit: all 11 parameters optimized in a single
fit_lse call (no Stage 1 → Stage 2 staging).

This is the cleanest test of: "does a true joint fit on this data
prefer σ_para < 0?".

Two data variants are supported via CLI flag:
  --variant all      : all 1ep + rep + para points (n≈301 with §6 corpus)
  --variant drop14m  : drop 14M non-para points (n≈258 with §6 corpus)

Initialization grid is configurable via flag.  Default grid brackets
σ_para around 0 ([-2, -1, 0, +1]) so the optimizer can land on either
sign.  Output: JSON with per-k optimum params + RMSE.
"""

import argparse
import glob
import json
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

from fit_joint_triple import (DELTA, SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA,  # noqa: E402
                                collect_pooled_triple, make_triple_forward)
from fit_lse import expand_grid, fit_lse  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────────────
# Grid catalog — each named grid initializes the 11 params differently
# ──────────────────────────────────────────────────────────────────────

GRID_CATALOG = {
    # default: brackets σ_para around 0, both signs of ρ_para
    "default": expand_grid({
        "e":             [0.0, 1.0],
        "a":             [3.0, 12.0],
        "b":             [5.0, 12.0],
        "alpha":         [0.15, 0.4],
        "beta":          [0.3, 0.5],
        "log_K_rep":     [10.0, 18.0],
        "rho_rep":       [-1.0, 0.0],
        "sigma_rep":     [-1.0, 0.0],
        "log_K_para":    [10.0, 14.0],
        "rho_para":      [-3.0, -1.0, 0.0],
        "sigma_para":    [-2.0, -1.0, 0.0, +1.0],
    }),
    # σ_para grid biased negative
    "para_neg": expand_grid({
        "e":             [0.0, 1.0],
        "a":             [3.0, 12.0],
        "b":             [5.0, 12.0],
        "alpha":         [0.15, 0.4],
        "beta":          [0.3, 0.5],
        "log_K_rep":     [10.0, 18.0],
        "rho_rep":       [-1.0, 0.0],
        "sigma_rep":     [-1.0, 0.0],
        "log_K_para":    [10.0, 14.0],
        "rho_para":      [-3.0, -2.0, -1.0],
        "sigma_para":    [-2.0, -1.5, -1.0, -0.5],
    }),
    # σ_para grid biased positive
    "para_pos": expand_grid({
        "e":             [0.0, 1.0],
        "a":             [3.0, 12.0],
        "b":             [5.0, 12.0],
        "alpha":         [0.15, 0.4],
        "beta":          [0.3, 0.5],
        "log_K_rep":     [10.0, 18.0],
        "rho_rep":       [-1.0, 0.0],
        "sigma_rep":     [-1.0, 0.0],
        "log_K_para":    [10.0, 14.0],
        "rho_para":      [-1.0, 0.0, +1.0],
        "sigma_para":    [+0.0, +0.5, +1.0, +1.5],
    }),
    # broad grid, sigma_para covers wide range
    "broad": expand_grid({
        "e":             [0.0, 1.0, 2.0],
        "a":             [3.0, 12.0],
        "b":             [5.0, 12.0],
        "alpha":         [0.15, 0.3, 0.45],
        "beta":          [0.3, 0.45],
        "log_K_rep":     [10.0, 14.0, 18.0],
        "rho_rep":       [-1.0, 0.0],
        "sigma_rep":     [-1.0, 0.0],
        "log_K_para":    [10.0, 14.0],
        "rho_para":      [-3.0, -1.0],
        "sigma_para":    [-2.0, -0.5, +0.5, +2.0],
    }),
    # tight log K, exhaustive (ρ, σ)
    "para_dense": expand_grid({
        "e":             [0.5],
        "a":             [7.0],
        "b":             [9.0],
        "alpha":         [0.2, 0.35],
        "beta":          [0.35, 0.5],
        "log_K_rep":     [12.0],
        "rho_rep":       [-0.5],
        "sigma_rep":     [-0.5],
        "log_K_para":    [10.0, 14.0],
        "rho_para":      [-3.0, -2.0, -1.0, 0.0, +1.0],
        "sigma_para":    [-2.0, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0],
    }),
}


# ──────────────────────────────────────────────────────────────────────
# Data filters
# ──────────────────────────────────────────────────────────────────────

def filter_data(data, mask):
    return {k: v[mask] for k, v in data.items()}


def get_data(variant="all", scale_min_para=0.5):
    """Return pooled data dict for the requested variant."""
    full = collect_pooled_triple(scale_min_para=scale_min_para,
                                   exclude_sizes_1ep=(),
                                   exclude_sizes_rep=(),
                                   exclude_sizes_para=())
    if variant == "all":
        return full
    if variant == "drop14m":
        keep = ~((full["tags"] == "14m") & (full["source"] != SOURCE_PARA))
        return filter_data(full, keep)
    raise ValueError(f"unknown variant: {variant}")


# ──────────────────────────────────────────────────────────────────────
# Fit + iterative drop
# ──────────────────────────────────────────────────────────────────────

def onego_fit(data, init_grid, delta=DELTA):
    """Single-stage 11-param joint fit on the full pooled data."""
    fwd = make_triple_forward(data["N"], data["D"], data["Dp"], data["source"])
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    return fit_lse(fwd, log_L, init_grid, delta=delta, verbose=False)


def topk_drop_sweep(data, init_params, k_values, delta=DELTA):
    """Same as fit_joint_triple.topk_drop_sweep but inline here."""
    fwd_full = make_triple_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    last = init_params
    keep = np.ones(len(data["L"]), dtype=bool)
    cumulative = []
    out = {}
    for k in sorted(set(k_values)):
        while len(cumulative) < k:
            grid = [{kk: float(vv) for kk, vv in last.items()}]
            fwd = make_triple_forward(
                data["N"][keep], data["D"][keep],
                data["Dp"][keep], data["source"][keep])
            log_L = torch.tensor(np.log(data["L"][keep]),
                                  dtype=torch.float64)
            res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
            with torch.no_grad():
                p_t = {kk: torch.tensor(v, dtype=torch.float64)
                       for kk, v in res["params"].items()}
                pred_full = fwd_full(p_t).numpy()
            resid = np.log(data["L"]) - pred_full
            ar = np.abs(resid); ar[~keep] = -1.0
            worst = int(np.argmax(ar))
            cumulative.append(worst); keep[worst] = False
            last = res["params"]
        grid = [{kk: float(vv) for kk, vv in last.items()}]
        fwd = make_triple_forward(
            data["N"][keep], data["D"][keep],
            data["Dp"][keep], data["source"][keep])
        log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
        res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
        last = res["params"]
        with torch.no_grad():
            p_t = {kk: torch.tensor(v, dtype=torch.float64)
                   for kk, v in res["params"].items()}
            pred_full = fwd_full(p_t).numpy()
        out[k] = dict(params=res["params"], pred_full=pred_full,
                       keep=keep.copy(), dropped=list(cumulative))
    return out


def _summary(data, params, pred_full, keep):
    log_L = np.log(data["L"]); resid = log_L - pred_full
    out = dict(rmse_full=float(np.sqrt(np.mean(resid ** 2))),
               rmse_kept=float(np.sqrt(np.mean(resid[keep] ** 2))))
    for tag, src_id in [("1ep", SOURCE_NONE), ("rep", SOURCE_REPEAT),
                         ("para", SOURCE_PARA)]:
        m = (data["source"] == src_id); mk = m & keep
        out[f"n_{tag}"] = int(m.sum()); out[f"n_{tag}_kept"] = int(mk.sum())
        out[f"rmse_{tag}"] = (float(np.sqrt(np.mean(resid[mk] ** 2)))
                                if mk.any() else float("nan"))
    return out


def serialize_params(p):
    return {k: float(v) for k, v in p.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["all", "drop14m"], default="all")
    ap.add_argument("--grid", choices=list(GRID_CATALOG.keys()),
                     default="default")
    ap.add_argument("--scale-min-para", type=float, default=0.5)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--canonical-k", type=int, default=15)
    args = ap.parse_args()

    print(f"# One-go joint triple fit")
    print(f"#   variant      = {args.variant}")
    print(f"#   init grid    = {args.grid} ({len(GRID_CATALOG[args.grid])} inits)")
    print(f"#   scale_min_p  = {args.scale_min_para}")
    print(f"#   canonical k  = {args.canonical_k}")

    data = get_data(args.variant, args.scale_min_para)
    n_per = {src: int((data["source"] == src).sum())
             for src in [SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA]}
    print(f"# Pooled n={len(data['L'])} "
          f"(1ep={n_per[SOURCE_NONE]}, "
          f"rep={n_per[SOURCE_REPEAT]}, para={n_per[SOURCE_PARA]})")

    print(f"\n[Stage 0] One-go 11-param joint fit "
          f"with {args.grid} grid...")
    res = onego_fit(data, GRID_CATALOG[args.grid])
    p0 = res["params"]
    rmse_in_sample = res.get("rmse_logL", float("nan"))
    print(f"  In-sample RMSE: {rmse_in_sample:.4f}")
    print(f"  Chinchilla: E={np.exp(p0['e']):.4f}  A={np.exp(p0['a']):.2f}  "
          f"B={np.exp(p0['b']):.2f}  α={p0['alpha']:.4f}  "
          f"β={p0['beta']:.4f}")
    print(f"  η_rep:      log K={p0['log_K_rep']:.2f}  "
          f"ρ={p0['rho_rep']:+.3f}  σ={p0['sigma_rep']:+.3f}")
    print(f"  η_para:     log K={p0['log_K_para']:.2f}  "
          f"ρ={p0['rho_para']:+.3f}  σ={p0['sigma_para']:+.3f}")

    K_VALUES = [0, 5, 10, 15, 20, 25, 30]
    print(f"\n[Stage 1] Iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(data, p0, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  {'E':>7s}  {'A':>7s}  "
          f"{'B':>9s}  {'α':>6s}  {'β':>6s}  | "
          f"{'lK_r':>6s} {'ρ_r':>6s} {'σ_r':>6s} | "
          f"{'lK_p':>6s} {'ρ_p':>6s} {'σ_p':>6s} | "
          f"{'1ep':>5s}  {'rep':>5s}  {'par':>5s}")
    sweep_records = []
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(data, r["params"], r["pred_full"], r["keep"])
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
        sweep_records.append(dict(
            k=k, n_kept=int(r["keep"].sum()),
            params=serialize_params(p),
            rmse=s,
        ))

    canonical = args.canonical_k
    rc = sweep[canonical]
    sc = _summary(data, rc["params"], rc["pred_full"], rc["keep"])
    pc = rc["params"]
    print(f"\n  Canonical k={canonical}:")
    print(f"    Chinchilla: E={np.exp(pc['e']):.4f}  A={np.exp(pc['a']):.2f}  "
          f"B={np.exp(pc['b']):.2f}  α={pc['alpha']:.4f}  "
          f"β={pc['beta']:.4f}")
    print(f"    η_rep:      log K={pc['log_K_rep']:.2f}  "
          f"ρ={pc['rho_rep']:+.3f}  σ={pc['sigma_rep']:+.3f}")
    print(f"    η_para:     log K={pc['log_K_para']:.2f}  "
          f"ρ={pc['rho_para']:+.3f}  σ={pc['sigma_para']:+.3f}")
    print(f"    RMSE — 1ep:{sc['rmse_1ep']:.4f}  rep:{sc['rmse_rep']:.4f}  "
          f"para:{sc['rmse_para']:.4f}  |  kept:{sc['rmse_kept']:.4f}")

    # Headline signs
    sigma_para_neg = pc["sigma_para"] < 0
    rho_para_neg = pc["rho_para"] < 0
    print(f"\n  >> σ_para sign: {'NEG' if sigma_para_neg else 'POS'} "
          f"({pc['sigma_para']:+.3f})")
    print(f"  >> ρ_para sign: {'NEG' if rho_para_neg else 'POS'} "
          f"({pc['rho_para']:+.3f})")

    if args.out_json:
        out_dict = dict(
            variant=args.variant, grid=args.grid,
            scale_min_para=args.scale_min_para,
            canonical_k=canonical,
            in_sample_rmse=rmse_in_sample,
            canonical=dict(
                params=serialize_params(pc),
                rmse=sc,
                sigma_para_neg=bool(sigma_para_neg),
                rho_para_neg=bool(rho_para_neg),
            ),
            sweep=sweep_records,
            n_pooled=int(len(data["L"])),
            n_per_source={"1ep": n_per[SOURCE_NONE],
                          "rep": n_per[SOURCE_REPEAT],
                          "para": n_per[SOURCE_PARA]},
        )
        with open(args.out_json, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"\nWrote {args.out_json}")

    return sweep, canonical


if __name__ == "__main__":
    main()
