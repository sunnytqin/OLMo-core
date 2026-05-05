"""
Sanity-plot: does the *chosen* repetition η form (Form B / exp-sat
with N-dependent R*) fit the paraphrase data?

  η = R*·(1 − e^{−x/R*}) / x,   x = D'/D,
  log R* = log K + ρ·log(D/N) + σ·log N.

Already in the joint table (see fit_eta_para.py); here we isolate it
and plot per-size to see whether the fit makes sense.

Output:
  fit_eta_para_formB.pdf — 4 panels (one per size) with per-point
  η_para (grey ΔL, white E_eff), the per-size Form-B fit (blue), and
  the repetition Form-B curve at the same (D/N, D'/D) for direct
  contrast (red dashed).
  fit_eta_para_formB_joint.pdf — joint Form-B fit on pooled data.
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

# Palatino styling — match paper_figures.py
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

from data import SIZES, TTP_RATIO, extract_paraphrase, load_with_para  # noqa: E402
from fit_eta import (FORMS, fit_form, leave_one_out, per_point_eta,  # noqa: E402
                     per_point_eta_diff)
from fit_eta_para import PARA_SIZES, get_joint_anchors  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 1e-3
SCALE_MIN = 0.5
FORM_NAME = "Muennighoff R*(N)"

# Repetition Form B R*(N) two-stage anchors (writeup §4.3 table):
REPEAT_FORM_B = dict(log_K=17.8, rho=-0.93, sigma=-0.69)


def _eta_formB(log_K, rho, sigma, D_arr, Dp_arr, N_arr):
    """η = R*(1 - e^{-x/R*}) / x with log R* = log K + ρ·log(D/N) + σ·log N."""
    R = np.exp(log_K + rho * np.log(D_arr / N_arr) + sigma * np.log(N_arr))
    x = Dp_arr / D_arr
    return R * (1.0 - np.exp(-x / R)) / x


def _Rstar(log_K, rho, sigma, D, N):
    return float(np.exp(log_K + rho * np.log(D / N) + sigma * np.log(N)))


def main():
    anchor = get_joint_anchors(verbose=True)
    B, beta = anchor["B"], anchor["beta"]
    E_j, A_j, alpha_j = anchor["E"], anchor["A"], anchor["alpha"]
    print(f"\nFitting Form B — η = R*(1−e^{{−x/R*}})/x,  "
          f"log R* = log K + ρ·log(D/N) + σ·log N — on paraphrase per size")

    per_size = {}
    for size in PARA_SIZES:
        N, datasets, parap = load_with_para(size)
        if not parap:
            continue
        s, D, K, Dp, L, L_1ep = extract_paraphrase(
            datasets, parap, N, scale_min=SCALE_MIN)
        if len(D) < 4:
            continue
        E_size = E_j + A_j / N ** alpha_j
        eta_pp_dL = per_point_eta_diff(D, Dp, L, L_1ep, B, beta)
        eta_pp_E  = per_point_eta(D, Dp, L, E_size, B, beta)
        res = fit_form(FORMS[FORM_NAME], D, Dp, N, L, E_size, B, beta,
                       delta=DELTA)
        loo = leave_one_out(FORMS[FORM_NAME], D, Dp, N, L, E_size, B, beta,
                             delta=DELTA, warm_init=res["params"])
        v = ~np.isnan(loo)
        loo_rmse = float(np.sqrt(np.mean(loo[v] ** 2))) if v.any() else np.nan
        p = res["params"]
        Rstar_at_1x = _Rstar(p["log_K"], p["rho"], p["sigma"],
                              TTP_RATIO * N, N)
        per_size[size] = dict(
            N=N, scale=s, D=D, Dp=Dp, K=K, L=L, L_1ep=L_1ep,
            eta_pp_dL=eta_pp_dL, eta_pp_E=eta_pp_E,
            params=p, rmse=res["rmse"], loo_rmse=loo_rmse,
            Rstar_1x=Rstar_at_1x,
        )
        print(f"  {size}  (N={N/1e6:.0f}M, n={len(D)}): "
              f"log K={p['log_K']:.2f}  ρ={p['rho']:+.2f}  "
              f"σ={p['sigma']:+.2f}  →  R*(1×)={Rstar_at_1x:.2g}  "
              f"RMSE={res['rmse']:.4f}  LOO={loo_rmse:.4f}")

    # ── Joint Form-B fit on pooled data ───────────────────────────
    print(f"\nJoint Form B on pooled paraphrase data:")
    from fit_eta_para import collect_paraphrase_all_sizes
    tags, Ns, scales, Ds, Ks, Dps, Ls, L_1eps = collect_paraphrase_all_sizes(
        SCALE_MIN)
    E_eff_arr = np.array([E_j + A_j / n ** alpha_j for n in Ns])
    res_joint = fit_form(FORMS[FORM_NAME], Ds, Dps, Ns, Ls, E_eff_arr,
                          B, beta, delta=DELTA)
    loo_joint = leave_one_out(FORMS[FORM_NAME], Ds, Dps, Ns, Ls, E_eff_arr,
                               B, beta, delta=DELTA,
                               warm_init=res_joint["params"])
    p_j = res_joint["params"]
    Rstar_30M_1x = _Rstar(p_j["log_K"], p_j["rho"], p_j["sigma"],
                           TTP_RATIO * 30e6, 30e6)
    print(f"  joint  log K={p_j['log_K']:.2f}  ρ={p_j['rho']:+.2f}  "
          f"σ={p_j['sigma']:+.2f}  RMSE={res_joint['rmse']:.4f}  "
          f"LOO={float(np.sqrt(np.mean(loo_joint**2))):.4f}")
    print(f"  → R*(30M, 1×) = {Rstar_30M_1x:.2g}  "
          f"(repetition Form B R*(30M, 1×) = "
          f"{_Rstar(REPEAT_FORM_B['log_K'], REPEAT_FORM_B['rho'], REPEAT_FORM_B['sigma'], TTP_RATIO * 30e6, 30e6):.2g})")
    print(f"  (repetition Form B: log K={REPEAT_FORM_B['log_K']}, "
          f"ρ={REPEAT_FORM_B['rho']}, σ={REPEAT_FORM_B['sigma']})")

    # ── Plot 4 panels (one per size) ──────────────────────────────
    sizes = sorted(per_size.keys(), key=lambda t: SIZES[t][0])
    fig, axes = plt.subplots(1, len(sizes),
                             figsize=(3.4 * len(sizes), 4.0),
                             squeeze=False, sharey=True)

    for col, size in enumerate(sizes):
        ax = axes[0, col]
        d = per_size[size]
        N = d["N"]
        D, Dp = d["D"], d["Dp"]
        x = Dp / D

        v_dL = ~np.isnan(d["eta_pp_dL"])
        ax.scatter(x[v_dL], d["eta_pp_dL"][v_dL], s=42, color="0.55",
                   edgecolors="k", linewidths=0.3, alpha=0.6, zorder=4,
                   label=r"per-point $\eta$ ($\Delta L$)")
        v_E = ~np.isnan(d["eta_pp_E"])
        ax.scatter(x[v_E], d["eta_pp_E"][v_E], s=42, color="white",
                   edgecolors="0.3", linewidths=0.6, alpha=0.85, zorder=5,
                   marker="s", label=r"per-point $\eta$ ($E_{\mathrm{eff}}$)")

        scales_here = sorted(set((D / (TTP_RATIO * N)).round(2).tolist()))
        cmap_b = plt.cm.Blues(np.linspace(0.45, 0.95, len(scales_here)))
        cmap_r = plt.cm.Reds(np.linspace(0.45, 0.95,  len(scales_here)))
        x_grid = np.geomspace(x.min() * 0.7, max(x.max() * 1.4, 5.0), 120)

        p = d["params"]
        for j, sc in enumerate(scales_here):
            D_val = sc * TTP_RATIO * N
            Dp_grid = x_grid * D_val
            D_arr = np.full_like(x_grid, D_val)
            N_arr = np.full_like(x_grid, float(N))

            eta_para = _eta_formB(p["log_K"], p["rho"], p["sigma"],
                                    D_arr, Dp_grid, N_arr)
            ax.plot(x_grid, eta_para, "-", color=cmap_b[j], linewidth=1.6,
                    alpha=0.95,
                    label=("para Form B" if j == 0 else None))

            eta_rep = _eta_formB(REPEAT_FORM_B["log_K"],
                                  REPEAT_FORM_B["rho"],
                                  REPEAT_FORM_B["sigma"],
                                  D_arr, Dp_grid, N_arr)
            ax.plot(x_grid, eta_rep, "--", color=cmap_r[j], linewidth=1.4,
                    alpha=0.85,
                    label=("repeat Form B" if j == 0 else None))

        ax.axhline(1.0, color="0.4", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.axvspan(x.min(), x.max(), color="0.92", alpha=0.4, zorder=0,
                   label="data range" if col == 0 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D'/D$")
        if col == 0:
            ax.set_ylabel(r"$\eta_{\mathrm{para}}$")
        title = (f"{size.upper()}  ($N={N/1e6:.0f}\\,$M)\n"
                 rf"$\log K={p['log_K']:.1f}$, "
                 rf"$\rho={p['rho']:+.2f}$, $\sigma={p['sigma']:+.2f}$")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.set_ylim(0.2, 3.0)
        ax.set_xlim(0.15, 5.0)
        if col == 0:
            ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9,
                      ncol=2, handlelength=1.5)

    fig.suptitle(
        r"Form B on paraphrase: $\eta = R^*(1 - e^{-x/R^*})/x$, "
        r"$\log R^* = \log K + \rho\,\log(D/N) + \sigma\,\log N$  "
        rf"— solid blue = paraphrase fit (per size); "
        rf"red dashed = repetition Form B "
        rf"($\log K={REPEAT_FORM_B['log_K']}$, "
        rf"$\rho={REPEAT_FORM_B['rho']}$, $\sigma={REPEAT_FORM_B['sigma']}$)",
        y=1.07)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fit_eta_para_formB.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Joint Form-B figure ────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(8.5, 5.0))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    for i, size in enumerate(sizes):
        d = per_size[size]
        v = ~np.isnan(d["eta_pp_dL"])
        ax.scatter(d["Dp"][v] / d["D"][v], d["eta_pp_dL"][v], s=55,
                   color=cmap[i], edgecolors="k", linewidths=0.3,
                   alpha=0.75, label=size.upper())
        N = d["N"]
        scales_here = sorted(set((d["D"] / (TTP_RATIO * N)).round(2).tolist()))
        for sc in scales_here:
            D_val = sc * TTP_RATIO * N
            x_grid = np.geomspace(0.2, 5.0, 80)
            Dp_grid = x_grid * D_val
            D_arr = np.full_like(x_grid, D_val)
            N_arr = np.full_like(x_grid, float(N))
            eta_curve = _eta_formB(p_j["log_K"], p_j["rho"], p_j["sigma"],
                                    D_arr, Dp_grid, N_arr)
            ax.plot(x_grid, eta_curve, "-", color=cmap[i],
                    linewidth=1.3, alpha=0.85)

    ax.axhline(1.0, color="0.4", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D'/D$")
    ax.set_ylabel(r"$\eta_{\mathrm{para}}$")
    ax.set_title(rf"Joint Form B on paraphrase: "
                 rf"$\log K={p_j['log_K']:.1f}$, "
                 rf"$\rho={p_j['rho']:+.2f}$, $\sigma={p_j['sigma']:+.2f}$  "
                 rf"(LOO={float(np.sqrt(np.mean(loo_joint**2))):.3f})")
    ax.legend(loc="upper right", ncol=2, fontsize=10)
    ax.grid(alpha=0.25, which="both")
    fig2.tight_layout()
    out2 = os.path.join(SCRIPT_DIR, "fit_eta_para_formB_joint.pdf")
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
