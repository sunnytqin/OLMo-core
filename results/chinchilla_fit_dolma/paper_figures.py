"""
Paper-ready figures for the multi-epoch Chinchilla writeup.

Uses URW P052 (Palatino-metric clone) since true Palatino isn't shipped.
Saves to ./paper/.

Figures
-------
F1  Joint 1-epoch Chinchilla fit across 7 sizes (data + curves +
    residuals + parity).
F2  η fit at the joint pool: per-point η coloured by N, with the
    chosen form (exp-sat) overlaid; residuals.
F3  Saturation R* vs N at fixed Chinchilla scales — the headline plot.
F4  η·D'/D vs epochs at scale 1× across 5 sizes (saturation curves).
F5  Marginal next-epoch return + epochs-to-saturate vs N (practitioner view).
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

# ── Palatino (URW P052 clone) setup ──────────────────────────────────
for _f in glob.glob('/usr/share/fonts/urw-base35/P052-*.otf'):
    font_manager.fontManager.addfont(_f)

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['P052', 'Palatino', 'TeX Gyre Pagella', 'serif'],
    'mathtext.fontset':  'cm',
    'font.size':         12,
    'axes.titlesize':    13,
    'axes.labelsize':    13,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'figure.titlesize':  14,
    'lines.linewidth':   1.8,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '-',
    'savefig.bbox':      'tight',
    'savefig.dpi':       300,
})

from data import OVERFIT_EXCLUDE, SIZES, TTP_RATIO, extract_1epoch  # noqa: E402
from data import extract_multi_epoch  # noqa: E402
from fit_chinchilla_joint import (collect_1epoch_all_sizes, fit_joint,  # noqa: E402
                                   predict, topk_residual_drop_sweep)
from fit_eta import (FORMS, collect_multi_epoch_all_sizes, fit_form,  # noqa: E402
                     per_point_eta_diff)
from fit_joint_all import fit_joint_all_sweep, collect_pooled  # noqa: E402

# ── Canonical pipeline: ONE-SHOT joint fit with k=15 residual drop ──
CANONICAL_K = 15
CANONICAL_FORM = "Muennighoff R*(N)"

PAPER_DIR = os.path.join(os.path.dirname(__file__), "paper")
os.makedirs(PAPER_DIR, exist_ok=True)

N_REF = 30e6  # reference N for R* parameterisation


def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.0f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def cmap_for(n):
    return plt.cm.viridis(np.linspace(0, 1, n))


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — joint Chinchilla 1-epoch fit
# ──────────────────────────────────────────────────────────────────────

def fig1_joint_chinchilla(one_shot_fit):
    """1-epoch panel of the one-shot joint fit (canonical pipeline)."""
    print("Figure 1: joint Chinchilla (one-shot k=15)")
    p = one_shot_fit["chinchilla"]
    data = one_shot_fit["data"]
    keep = one_shot_fit["keep"]
    is_1ep = ~data["is_multi"]

    # Slice to 1-epoch points for this figure
    tags = data["tags"][is_1ep]
    N_arr = data["N"][is_1ep]
    D_arr = data["D"][is_1ep]
    L_arr = data["L"][is_1ep]
    keep_1ep = keep[is_1ep]
    pred = predict(N_arr, D_arr, p)
    resid = np.log(L_arr) - np.log(pred)
    sizes = sorted(set(tags.tolist()), key=lambda t: SIZES[t][0])
    colors = {t: c for t, c in zip(sizes, cmap_for(len(sizes)))}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax_data, ax_resid, ax_par = axes

    # Panel A: L vs D, fit curves per N
    D_smooth = np.geomspace(D_arr.min() * 0.5, D_arr.max() * 2, 200)
    for tag in sizes:
        m = tags == tag
        ax_data.scatter(D_arr[m & keep_1ep], L_arr[m & keep_1ep], s=42,
                        color=colors[tag], edgecolors="k", linewidths=0.4,
                        zorder=5, label=f"{tag}")
        ax_data.scatter(D_arr[m & ~keep_1ep], L_arr[m & ~keep_1ep], s=70,
                        facecolors='none', edgecolors=colors[tag],
                        linewidths=1.2, zorder=4)
        L_curve = predict(SIZES[tag][0], D_smooth, p)
        ax_data.plot(D_smooth, L_curve, '-', color=colors[tag],
                     alpha=0.8, linewidth=1.5)
    ax_data.set_xscale('log', base=2)
    ax_data.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_data.set_xlabel(r"training tokens $D$")
    ax_data.set_ylabel(r"validation loss $L$")
    ax_data.set_title("(a) one-shot joint fit, 1-epoch panel")
    ax_data.legend(title=r"$N$", loc="upper right", ncol=2,
                   columnspacing=1.0, handletextpad=0.4)
    ax_data.text(0.04, 0.05,
                 rf"$\beta={p['beta']:.3f}$, $\alpha={p['alpha']:.3f}$" + "\n"
                 + rf"open $\circ$ = dropped (top-$k$, $k={CANONICAL_K}$)",
                 transform=ax_data.transAxes,
                 va='bottom', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    # Panel B: residuals
    for tag in sizes:
        m = tags == tag
        ax_resid.scatter(D_arr[m & keep_1ep], resid[m & keep_1ep], s=42,
                         color=colors[tag], edgecolors="k", linewidths=0.4)
        ax_resid.scatter(D_arr[m & ~keep_1ep], resid[m & ~keep_1ep], s=70,
                         facecolors='none', edgecolors=colors[tag],
                         linewidths=1.2)
    ax_resid.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_resid.set_xscale('log', base=2)
    ax_resid.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_resid.set_xlabel(r"training tokens $D$")
    ax_resid.set_ylabel(r"$\log L - \widehat{\log L}$")
    ax_resid.set_title("(b) residuals")

    # Panel C: parity (only kept 1-ep points)
    L_kept = L_arr[keep_1ep]
    pred_kept = pred[keep_1ep]
    for tag in sizes:
        m = (tags == tag) & keep_1ep
        ax_par.scatter(pred[m], L_arr[m],
                       s=42, color=colors[tag], edgecolors='k', linewidths=0.4)
    lo = min(L_kept.min(), pred_kept.min()) * 0.97
    hi = max(L_kept.max(), pred_kept.max()) * 1.03
    ax_par.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    ax_par.set_xlabel(r"predicted $L$")
    ax_par.set_ylabel(r"observed $L$")
    rmse = float(np.sqrt(np.mean(resid[keep_1ep] ** 2)))
    r2 = (1 - np.sum(resid[keep_1ep] ** 2) /
          np.sum((np.log(L_arr[keep_1ep]) - np.log(L_arr[keep_1ep]).mean()) ** 2))
    ax_par.set_title(f"(c) parity (1-ep only)   RMSE={rmse:.3f}, $R^2$={r2:.3f}")

    fig.tight_layout()
    out = os.path.join(PAPER_DIR, "fig1_joint_chinchilla.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")
    return p


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — η fit (joint pool, exp-sat with R*(N))
# ──────────────────────────────────────────────────────────────────────

def _predict_one_shot(N, D, Dp, chinch, eta_p):
    """L pred from one-shot fit (Muennighoff R*(N) form)."""
    log_R = (eta_p["log_K"] + eta_p["rho"] * np.log(D / N)
             + eta_p["sigma"] * np.log(N))
    R = np.exp(log_R)
    x = Dp / D
    eta = np.where(Dp > 0, R * (1 - np.exp(-x / R)) / np.where(Dp > 0, x, 1.0),
                   0.0)
    D_eff = D + eta * Dp
    return chinch["E"] + chinch["A"] / N ** chinch["alpha"] + chinch["B"] / D_eff ** chinch["beta"]


def fig2_eta_joint(one_shot_fit):
    """η panel of the one-shot fit: per-point η coloured by N + fitted curves."""
    print("Figure 2: η joint (one-shot k=15)")
    p = one_shot_fit["chinchilla"]
    eta_p = one_shot_fit["eta_params"]
    data = one_shot_fit["data"]
    keep = one_shot_fit["keep"]
    log_K = eta_p["log_K"]
    rho = eta_p["rho"]
    sigma = eta_p["sigma"]

    # per-point η on multi-ep points via ΔL (uses observed 1-ep loss at same scale)
    tags, Ns, scales, Ds, eps, Dps, Ls, L_1eps = collect_multi_epoch_all_sizes(0.5)
    eta_pp = per_point_eta_diff(Ds, Dps, Ls, L_1eps, p["B"], p["beta"])
    valid = ~np.isnan(eta_pp) & (eta_pp >= 0) & (eta_pp <= 5)

    # multi-epoch residuals from the one-shot fit, restricted to kept points
    is_multi = data["is_multi"]
    keep_multi = is_multi & keep
    resid_multi = one_shot_fit["resid"][keep_multi]
    D_multi = data["D"][keep_multi]
    Dp_multi = data["Dp"][keep_multi]
    tags_multi = data["tags"][keep_multi]

    sizes = sorted(set(data["tags"].tolist()), key=lambda t: SIZES[t][0])
    colors = {t: c for t, c in zip(sizes, cmap_for(len(sizes)))}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax_eta, ax_res = axes

    # Panel A: per-point η coloured by N + fitted curves at scale 1×
    for tag in sizes:
        m = (tags == tag) & valid
        if not m.any():
            continue
        ax_eta.scatter(Dps[m] / Ds[m], eta_pp[m], s=36, color=colors[tag],
                       edgecolors='k', linewidths=0.3, alpha=0.8,
                       label=tag)

    x_smooth = np.geomspace(0.3, 200, 200)
    for tag in sizes:
        N = SIZES[tag][0]
        # at scale 1×: D/N = TTP_RATIO
        log_R = log_K + rho * np.log(TTP_RATIO) + sigma * np.log(N)
        R_star = np.exp(log_R)
        eta_curve = R_star * (1 - np.exp(-x_smooth / R_star)) / x_smooth
        ax_eta.plot(x_smooth, eta_curve, '-', color=colors[tag],
                    alpha=0.8, linewidth=1.4)

    ax_eta.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax_eta.set_xscale('log')
    ax_eta.set_yscale('log')
    ax_eta.set_xlabel(r"$D'/D$  (additional / fresh tokens)")
    ax_eta.set_ylabel(r"$\eta$  (effective-token multiplier)")
    ax_eta.set_title(r"(a) $\eta$ per-point + exp-sat fit at scale 1$\times$")
    ax_eta.legend(title=r"$N$", loc="lower left", ncol=2,
                  columnspacing=1.0, handletextpad=0.4)
    rmse_multi = float(np.sqrt(np.mean(resid_multi ** 2)))
    ax_eta.text(0.04, 0.96,
                rf"$\log K = {log_K:.2f}$" + "\n"
                + rf"$\rho = {rho:.2f}$,  $\sigma = {sigma:.2f}$" + "\n"
                + rf"multi-ep RMSE = {rmse_multi:.3f}",
                transform=ax_eta.transAxes, va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    # Panel B: residuals on kept multi-epoch points (one-shot fit)
    for tag in sizes:
        m = tags_multi == tag
        if not m.any():
            continue
        ax_res.scatter(Dp_multi[m] / D_multi[m], resid_multi[m], s=36,
                       color=colors[tag], edgecolors='k', linewidths=0.3,
                       alpha=0.85, label=tag)
    ax_res.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax_res.set_xscale('log')
    ax_res.set_xlabel(r"$D'/D$")
    ax_res.set_ylabel(r"$\log L - \widehat{\log L}$")
    ax_res.set_title("(b) η fit residuals (kept multi-epoch points)")

    fig.tight_layout()
    out = os.path.join(PAPER_DIR, "fig2_eta_joint.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 3 — R* vs N (saturation, the headline plot)
# ──────────────────────────────────────────────────────────────────────

def Rstar_KN(D, N, log_K, rho, sigma):
    return np.exp(log_K + rho * np.log(D / N) + sigma * np.log(N))


def fig3_Rstar_vs_N(one_shot_fit, scales=(0.5, 1.0, 2.0, 4.0)):
    print("Figure 3: R* vs N")
    eta_p = one_shot_fit["eta_params"]
    log_K = eta_p["log_K"]
    rho = eta_p["rho"]
    sigma = eta_p["sigma"]

    sizes = sorted(SIZES, key=lambda t: SIZES[t][0])
    Ns = np.array([SIZES[s][0] for s in sizes])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(scales)))
    for j, scale in enumerate(scales):
        D_arr = scale * TTP_RATIO * Ns
        R = Rstar_KN(D_arr, Ns, log_K, rho, sigma)
        ax.plot(Ns, R, 'o-', color=cmap[j], markersize=8,
                label=rf"scale = ${scale}\times$")
    for s, N in zip(sizes, Ns):
        D = 1.0 * TTP_RATIO * N
        R = Rstar_KN(D, N, log_K, rho, sigma)
        ax.annotate(s, (N, R), textcoords="offset points",
                    xytext=(8, 4), fontsize=10, color="0.4")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"model parameters $N$")
    ax.set_ylabel(r"saturation $R^{*} = K \cdot (D/N)^{\rho} \cdot N^{\sigma}$")
    ax.set_title(r"$R^{*}$ vs $N$  —  larger models saturate sooner at fixed scale")
    ax.legend(loc='best')
    ax.text(0.04, 0.04,
            rf"joint exp-sat fit: $\log K={log_K:.2f}$, "
            rf"$\rho={rho:.2f}$, $\sigma={sigma:.2f}$",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    fig.tight_layout()
    out = os.path.join(PAPER_DIR, "fig3_Rstar_vs_N.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 4 — saturation curves: η·D'/D vs epochs, by size
# ──────────────────────────────────────────────────────────────────────

def fig4_saturation_curves(one_shot_fit, scale=1.0):
    print(f"Figure 4: saturation curves at scale {scale}×")
    eta_p = one_shot_fit["eta_params"]
    log_K = eta_p["log_K"]
    rho = eta_p["rho"]
    sigma = eta_p["sigma"]

    sizes_with_data = ["14m", "30m", "60m", "190m", "370m"]
    sizes = sorted(sizes_with_data, key=lambda t: SIZES[t][0])
    colors = {t: c for t, c in zip(sizes, cmap_for(len(sizes)))}

    epochs = np.geomspace(1.5, 256, 200)
    x = epochs - 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_eta, ax_xtra = axes

    for tag in sizes:
        N = SIZES[tag][0]
        D = scale * TTP_RATIO * N
        R = Rstar_KN(D, N, log_K, rho, sigma)
        eta = R * (1 - np.exp(-x / R)) / x
        ax_eta.plot(epochs, eta, '-', color=colors[tag], linewidth=2.0,
                    label=f"{tag}  $R^*$={R:.1f}")
        ax_xtra.plot(epochs, eta * x, '-', color=colors[tag], linewidth=2.0,
                     label=f"{tag}")
        ax_xtra.axhline(R, color=colors[tag], linestyle=':', linewidth=1.0,
                        alpha=0.5)

    ax_eta.set_xscale('log', base=2)
    ax_eta.set_yscale('log')
    ax_eta.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax_eta.set_xlabel("epochs")
    ax_eta.set_ylabel(r"$\eta$")
    ax_eta.set_title(rf"(a) $\eta$ vs epochs at scale ${scale}\times$")
    ax_eta.legend(loc='lower left', ncol=1)

    ax_xtra.set_xscale('log', base=2)
    ax_xtra.set_xlabel("epochs")
    ax_xtra.set_ylabel(r"$\eta \cdot D'/D$  (extra fresh-equivalent tokens / $D$)")
    ax_xtra.set_title(rf"(b) saturation: dotted = $R^*$ asymptote")
    ax_xtra.legend(loc='lower right')

    fig.tight_layout()
    out = os.path.join(PAPER_DIR, "fig4_saturation_curves.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 5 — practitioner view: marginal return + epochs-to-saturate
# ──────────────────────────────────────────────────────────────────────

def fig5_marginal_and_halflife(one_shot_fit, scale=1.0):
    """Two practitioner-focused panels:
       (a) Marginal next-epoch return  m(x) = ∂(η·D'/D)/∂(D'/D) = exp(−x/R*)
           — the value of one extra epoch in fresh-token-equivalents.
       (b) Epochs to reach 50%/90% of R* vs N, at scale 1×.
    """
    print(f"Figure 5: marginal return + half-saturation vs N (scale {scale}×)")
    eta_p = one_shot_fit["eta_params"]
    log_K = eta_p["log_K"]
    rho = eta_p["rho"]
    sigma = eta_p["sigma"]

    sizes_curve = ["14m", "30m", "60m", "190m", "370m"]
    sizes_curve = sorted(sizes_curve, key=lambda t: SIZES[t][0])
    colors = {t: c for t, c in zip(sizes_curve, cmap_for(len(sizes_curve)))}

    epochs = np.geomspace(1.5, 256, 200)
    x = epochs - 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_marg, ax_half = axes

    # Panel A: marginal next-epoch return exp(-x/R*) vs epochs
    for tag in sizes_curve:
        N = SIZES[tag][0]
        D = scale * TTP_RATIO * N
        R = Rstar_KN(D, N, log_K, rho, sigma)
        m = np.exp(-x / R)
        ax_marg.plot(epochs, m, '-', color=colors[tag], linewidth=2.0,
                     label=f"{tag}  $R^*$={R:.1f}")
    for frac, lbl in [(0.5, "50%"), (0.1, "10%")]:
        ax_marg.axhline(frac, color="gray", linestyle=":", linewidth=1.0)
        ax_marg.text(epochs[-1] * 0.95, frac * 1.06, lbl,
                     ha="right", va="bottom", fontsize=9, color="0.4")
    ax_marg.set_xscale('log', base=2)
    ax_marg.set_yscale('log')
    ax_marg.set_xlabel("epochs")
    ax_marg.set_ylabel(r"marginal next-epoch return $\;e^{-x/R^{*}}$")
    ax_marg.set_title(rf"(a) value of the next epoch (scale ${scale}\times$)")
    ax_marg.legend(loc='lower left', ncol=1)

    # Panel B: epochs to reach {50%, 90%} of R* vs N, at multiple scales
    sizes_all = sorted(SIZES, key=lambda t: SIZES[t][0])
    Ns = np.array([SIZES[s][0] for s in sizes_all])
    scales_b = (0.5, 1.0, 4.0)
    cmap_b = plt.cm.plasma(np.linspace(0.15, 0.85, len(scales_b)))
    for j, sc in enumerate(scales_b):
        Ds = sc * TTP_RATIO * Ns
        Rs = Rstar_KN(Ds, Ns, log_K, rho, sigma)
        ep_50 = 1 + Rs * np.log(2)
        ep_90 = 1 + Rs * np.log(10)
        ax_half.plot(Ns, ep_50, 'o-', color=cmap_b[j], markersize=7,
                     label=rf"50% of $R^*$, scale ${sc}\times$")
        ax_half.plot(Ns, ep_90, 's--', color=cmap_b[j], markersize=7,
                     alpha=0.85,
                     label=rf"90% of $R^*$, scale ${sc}\times$")
    ax_half.set_xscale('log')
    ax_half.set_yscale('log')
    ax_half.set_xlabel(r"model parameters $N$")
    ax_half.set_ylabel(r"epochs to reach fraction of $R^{*}$")
    ax_half.set_title(r"(b) saturation budget consumed vs $N$")
    ax_half.legend(loc='best', fontsize=9, ncol=1)

    fig.tight_layout()
    out = os.path.join(PAPER_DIR, "fig5_marginal_halflife.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running canonical one-shot fit: form='{CANONICAL_FORM}', k={CANONICAL_K}")
    sweep = fit_joint_all_sweep(form_name=CANONICAL_FORM,
                                k_values=(CANONICAL_K,))
    one_shot_fit = sweep[CANONICAL_K]
    ch = one_shot_fit["chinchilla"]; ep = one_shot_fit["eta_params"]
    print(f"  Chinchilla: E={ch['E']:.4f} A={ch['A']:.2f} B={ch['B']:.0f} "
          f"α={ch['alpha']:.4f} β={ch['beta']:.4f}")
    print(f"  η         : log K={ep['log_K']:.3f} ρ={ep['rho']:.3f} σ={ep['sigma']:.3f}")
    print(f"  RMSE 1ep={one_shot_fit['rmse_1ep']:.3f} multi={one_shot_fit['rmse_multi']:.3f}\n")

    fig1_joint_chinchilla(one_shot_fit)
    fig2_eta_joint(one_shot_fit)
    fig3_Rstar_vs_N(one_shot_fit)
    fig4_saturation_curves(one_shot_fit)
    fig5_marginal_and_halflife(one_shot_fit)
    print("\nDone. Figures in:", PAPER_DIR)
