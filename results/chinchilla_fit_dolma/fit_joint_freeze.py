"""
Frozen-Chinchilla fit: η_para only.

Stage 1 is *not* re-fitted. The shared Chinchilla parameters are pinned to
the writeup_final / one-shot rep+1ep headline values:

    E = 0.050,  A = 31,  B = 16,539,  α = 0.137,  β = 0.436

(Plus η_rep set to its writeup_final values — only used for completeness;
this script fits on paraphrase data only, so η_rep never enters the loss.)

Functional form (paraphrase points only):

    L(N, D, D'; para) = E + A/N^α + B / (D + η_para(D, D'; N) · D')^β
    η_para(D, D'; N) = R*_para · (1 − e^{−x/R*_para}) / x,    x = D'/D
    log R*_para     = log K_para + ρ_para · log(D/N) + σ_para · log N

Procedure:
    Stage 2.  Fit (log K_para, ρ_para, σ_para) on paraphrase data with all
              other parameters held at their frozen values.
    Stage 3.  Iterative residual-greedy drop sweep on paraphrase points,
              warm-starting η_para at every k.

Output:
    fit_joint_freeze.json — canonical-k summary.
    fit_joint_freeze.pdf  — diagnostic plot.
"""

import glob
import json
import os
import sys
from typing import Dict

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
from fit_joint_triple import (SOURCE_NONE, SOURCE_PARA, SOURCE_REPEAT,  # noqa: E402
                                PARA_GRID, fmt_tokens,
                                make_triple_forward)
from fit_lse import fit_lse  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.1

# Sizes excluded from the paraphrase pool (14M is excluded by default — it
# extrapolates poorly in the joint repetition fits as well).
EXCLUDE_SIZES = {"14m"}

# ──────────────────────────────────────────────────────────────────────
# Frozen Stage-1 parameters (writeup_final / one-shot rep+1ep, k=15)
# ──────────────────────────────────────────────────────────────────────

FROZEN_E     = 0.050
FROZEN_A     = 31.0
FROZEN_B     = 16539.0
FROZEN_ALPHA = 0.137
FROZEN_BETA  = 0.436

# η_rep is unused (we fit on paraphrase only), but the triple-forward needs
# values to evaluate. Pin to writeup_final reference for transparency.
FROZEN_LOG_K_REP = 10.32
FROZEN_RHO_REP   = -0.270
FROZEN_SIGMA_REP = -0.388

FROZEN_FULL: Dict[str, float] = {
    "e":         float(np.log(FROZEN_E)),
    "a":         float(np.log(FROZEN_A)),
    "b":         float(np.log(FROZEN_B)),
    "alpha":     FROZEN_ALPHA,
    "beta":      FROZEN_BETA,
    "log_K_rep": FROZEN_LOG_K_REP,
    "rho_rep":   FROZEN_RHO_REP,
    "sigma_rep": FROZEN_SIGMA_REP,
}


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def collect_para_only():
    """Pool paraphrase points across all sizes that have parap_datasets,
    skipping any size in EXCLUDE_SIZES."""
    tags, Ns, Ds, Dps, Ls, src = [], [], [], [], [], []
    for size in SIZES:
        if size in EXCLUDE_SIZES:
            continue
        N, datasets, parap = load_with_para(size)
        if not parap:
            continue
        _, D_, _, Dp_, L_, _ = extract_paraphrase(
            datasets, parap, N, scale_min=0.0)
        for d, dp, l in zip(D_, Dp_, L_):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
            Ls.append(l); src.append(SOURCE_PARA)
    return dict(
        tags=np.array(tags),
        N=np.array(Ns, dtype=np.float64),
        D=np.array(Ds, dtype=np.float64),
        Dp=np.array(Dps, dtype=np.float64),
        L=np.array(Ls, dtype=np.float64),
        source=np.array(src, dtype=np.int64),
    )


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────

def _wrap_fixed(forward_fn, fixed_params):
    fixed_t = {k: torch.tensor(float(v), dtype=torch.float64)
               for k, v in fixed_params.items()}

    def fwd(p):
        return forward_fn({**p, **fixed_t})

    return fwd


def stage2_para_only(data, fixed_params, delta=DELTA):
    """Fit only η_para (3 params); all 8 frozen params held fixed."""
    base = make_triple_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    fwd = _wrap_fixed(base, fixed_params)
    init_grid = list(PARA_GRID)
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    return fit_lse(fwd, log_L, init_grid, delta=delta, verbose=False)


def topk_drop_sweep(data, init_para_params, fixed_params, k_values,
                     delta=DELTA):
    """Iterative residual drop with η_para refit at each step."""
    base_full = make_triple_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    fwd_full = _wrap_fixed(base_full, fixed_params)
    last = dict(init_para_params)
    keep = np.ones(len(data["L"]), dtype=bool)
    cumulative = []
    out = {}
    for k in sorted(set(k_values)):
        while len(cumulative) < k:
            grid = [{kk: float(vv) for kk, vv in last.items()}]
            base = make_triple_forward(
                data["N"][keep], data["D"][keep],
                data["Dp"][keep], data["source"][keep])
            fwd = _wrap_fixed(base, fixed_params)
            log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
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
        # Final fit at this k
        grid = [{kk: float(vv) for kk, vv in last.items()}]
        base = make_triple_forward(
            data["N"][keep], data["D"][keep],
            data["Dp"][keep], data["source"][keep])
        fwd = _wrap_fixed(base, fixed_params)
        log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
        res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
        last = res["params"]
        with torch.no_grad():
            p_t = {kk: torch.tensor(v, dtype=torch.float64)
                   for kk, v in res["params"].items()}
            pred_full = fwd_full(p_t).numpy()
        full_params = {**res["params"],
                        **{k: float(v) for k, v in fixed_params.items()}}
        out[k] = dict(params=full_params,
                       pred_full=pred_full,
                       keep=keep.copy(),
                       dropped=list(cumulative))
    return out


def _summary(data, params, pred_full, keep):
    log_L = np.log(data["L"])
    resid = log_L - pred_full
    out = dict(rmse_full=float(np.sqrt(np.mean(resid ** 2))),
               rmse_kept=float(np.sqrt(np.mean(resid[keep] ** 2))))
    m = (data["source"] == SOURCE_PARA)
    mk = m & keep
    out["n_para"] = int(m.sum())
    out["n_para_kept"] = int(mk.sum())
    out["rmse_para"] = (float(np.sqrt(np.mean(resid[mk] ** 2)))
                          if mk.any() else float("nan"))
    return out


def pick_canonical_k(sweep, data, k_values, tol=0.001):
    """First k whose para-RMSE plateaus vs the previous k (Δ < tol)."""
    rmses = []
    for k in k_values:
        s = _summary(data, sweep[k]["params"],
                     sweep[k]["pred_full"], sweep[k]["keep"])
        rmses.append(s["rmse_para"])
    for i in range(1, len(rmses)):
        if abs(rmses[i] - rmses[i - 1]) < tol:
            return k_values[i]
    return k_values[-1]


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def plot_diagnostic(data, pred_full, keep, dropped, params, summary, path):
    sizes = sorted(set(data["tags"].tolist()), key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sizes)))
    color_of = {s: cmap[i] for i, s in enumerate(sizes)}

    log_L = np.log(data["L"])
    resid = log_L - pred_full

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_data, ax_res, ax_par = axes

    for tag in sizes:
        m = data["tags"] == tag
        c = color_of[tag]
        ax_data.scatter(data["D"][m] + data["Dp"][m], data["L"][m],
                        s=50, color=c, edgecolors="k", linewidths=0.5,
                        marker="s", alpha=0.85, zorder=3, label=tag)
        ax_res.scatter(data["D"][m], resid[m], s=40, color=c,
                       edgecolors="k", linewidths=0.3, marker="s")
        ax_par.scatter(np.exp(pred_full[m]), data["L"][m], s=42, color=c,
                       edgecolors="k", linewidths=0.3)

    if len(dropped):
        dmask = np.zeros(len(data["L"]), dtype=bool); dmask[dropped] = True
        ax_data.scatter(data["D"][dmask] + data["Dp"][dmask],
                        data["L"][dmask], s=170, facecolors="none",
                        edgecolors="red", linewidths=1.6, zorder=10,
                        label="dropped")

    ax_data.set_xscale("log", base=2)
    ax_data.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_data.set_xlabel(r"Effective tokens $D + D'$")
    ax_data.set_ylabel("Validation loss")
    ax_data.legend(loc="upper right", ncol=2, fontsize=9)
    ax_data.grid(alpha=0.3)
    p = params
    ax_data.set_title(
        rf"Frozen Chinchilla: $E={np.exp(p['e']):.3f}$, "
        rf"$A={np.exp(p['a']):.0f}$, "
        rf"$B={np.exp(p['b']):.0f}$, $\alpha={p['alpha']:.3f}$, "
        rf"$\beta={p['beta']:.3f}$" + "\n"
        rf"η_para (fit): $\log K={p['log_K_para']:.2f}$, "
        rf"$\rho={p['rho_para']:+.3f}$, "
        rf"$\sigma={p['sigma_para']:+.3f}$",
        fontsize=11)

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_res.set_xlabel(r"$D$")
    ax_res.set_ylabel(r"residual ($\log L$)")
    ax_res.set_title(rf"Paraphrase residuals "
                     rf"(kept RMSE: {summary['rmse_para']:.3f})")
    ax_res.grid(alpha=0.3)

    lo = min(np.exp(pred_full).min(), data["L"].min()) * 0.95
    hi = max(np.exp(pred_full).max(), data["L"].max()) * 1.05
    ax_par.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6)
    ax_par.set_xlabel("Predicted L")
    ax_par.set_ylabel("Observed L")
    ax_par.set_title("Parity")
    ax_par.grid(alpha=0.3)

    fig.suptitle(
        "Frozen-Chinchilla fit: η_para fitted on paraphrase data only "
        "(□ para)",
        fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _print_params(label, p):
    print(f"  {label}")
    print(f"    Chinchilla: E={np.exp(p['e']):.4f}  A={np.exp(p['a']):.2f}  "
          f"B={np.exp(p['b']):.2f}  α={p['alpha']:.4f}  β={p['beta']:.4f}")
    if "log_K_rep" in p:
        print(f"    η_rep:      log K={p['log_K_rep']:.2f}  "
              f"ρ={p['rho_rep']:+.3f}  σ={p['sigma_rep']:+.3f}  (frozen)")
    if "log_K_para" in p:
        print(f"    η_para:     log K={p['log_K_para']:.2f}  "
              f"ρ={p['rho_para']:+.3f}  σ={p['sigma_para']:+.3f}")


def main():
    print("=" * 96)
    print("Frozen-Chinchilla fit: paraphrase-only η_para over pinned "
          "Stage-1 params")
    print("=" * 96)

    data = collect_para_only()
    excl = sorted(EXCLUDE_SIZES) if EXCLUDE_SIZES else "(none)"
    print(f"\nPooled (paraphrase only): n_total={len(data['L'])}  "
          f"[excluded sizes: {excl}]")
    sizes_seen = sorted(set(data["tags"].tolist()),
                        key=lambda t: SIZES[t][0])
    counts = {t: int((data["tags"] == t).sum()) for t in sizes_seen}
    print(f"  per-size counts: {counts}")

    print("\n[Stage 1] FROZEN — using writeup_final headline values:")
    print(f"  E={FROZEN_E}, A={FROZEN_A}, B={FROZEN_B}, "
          f"α={FROZEN_ALPHA}, β={FROZEN_BETA}")
    print(f"  η_rep (frozen, unused): logK={FROZEN_LOG_K_REP}, "
          f"ρ={FROZEN_RHO_REP}, σ={FROZEN_SIGMA_REP}")

    fixed_params = dict(FROZEN_FULL)

    print(f"\n[Stage 2] Fitting η_para only "
          f"(8 frozen params held fixed, 3 trainable)...")
    res2 = stage2_para_only(data, fixed_params)
    para_init = res2["params"]
    p2 = {**para_init, **fixed_params}
    _print_params("(stage 2 — η_para over frozen Stage 1)", p2)

    K_VALUES = [0, 2, 5, 8, 10, 12, 15]
    print(f"\n[Stage 3] Iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(data, para_init, fixed_params, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  | "
          f"{'lK_par':>7s} {'ρ_par':>7s} {'σ_par':>7s} | "
          f"{'para':>5s}")
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(data, r["params"], r["pred_full"], r["keep"])
        p = r["params"]
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  | "
              f"{p['log_K_para']:>7.2f} {p['rho_para']:>+7.3f} "
              f"{p['sigma_para']:>+7.3f} | "
              f"{s['rmse_para']:>5.3f}")

    canonical = pick_canonical_k(sweep, data, K_VALUES, tol=0.001)
    print(f"\n  Canonical k (first para-RMSE plateau): k = {canonical}")
    rc = sweep[canonical]
    sc = _summary(data, rc["params"], rc["pred_full"], rc["keep"])
    _print_params(f"(canonical k={canonical})", rc["params"])
    print(f"    para RMSE (kept): {sc['rmse_para']:.4f}  |  "
          f"all-points RMSE: {sc['rmse_full']:.4f}")

    out_json = {
        "method": "frozen-chinchilla, paraphrase-only η_para fit",
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
        "n_total": int(len(data["L"])),
        "per_size_counts": counts,
    }
    json_path = os.path.join(SCRIPT_DIR, "fit_joint_freeze.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved {json_path}")

    plot_diagnostic(
        data, rc["pred_full"], rc["keep"], rc["dropped"],
        rc["params"], sc,
        path=os.path.join(SCRIPT_DIR, "fit_joint_freeze.pdf"))

    return sweep, canonical


if __name__ == "__main__":
    main()
