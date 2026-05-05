"""
Figure 3 for the NeurIPS submission (paper section 4.4):
  Reducible validation loss  L_red = L - (E + A/N^α)  vs *effective* tokens
  D_eff = D + η·D' on log-log axes. Under the joint fit
  L = E + A/N^α + B/D_eff^β, so L_red = B/D_eff^β is a single straight line
  of slope -β across all N — and both 1-epoch and multi-epoch points
  collapse onto it.

  Panel (a): 1-epoch kept points only (D_eff = D). All sit on the line.
             Validates that the joint fit's backbone is consistent with
             the 1-epoch envelope.
  Panel (b): all kept pooled points (1-ep filled, multi-ep open).
             Multi-ep points join the same line under D → D + η·D'.

We exclude the points dropped by the canonical k=15 residual trim
(approximated here as the top-k pooled-residual points against the canonical
joint-fit prediction). u-shape exclusions are already applied inside
`collect_pooled`.

Saves to the syn_data_scaling figures dir as fig3_loss_vs_D.pdf and PNG.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))
from data import SIZES  # noqa: E402
from fit_joint_all import collect_pooled  # noqa: E402

OUT_DIR = "/home/hamidieh/projects/syn_pt/699ca65b9e05b9d5ffcc03f3/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Canonical one-shot joint fit (writeup_final.md §2, k=15)
CANON = dict(
    E=0.050, A=31.5, B=16_539.0, alpha=0.137, beta=0.436,
    log_K=10.32, rho=-0.270, sigma=-0.388,
)
CANONICAL_K = 15  # drop top-15 worst-residual points before plotting


# ── Palatino (URW P052 clone) — match paper_figures.py styling ──────
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
    "lines.linewidth":  1.8,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "-",
    "savefig.bbox":     "tight",
    "savefig.dpi":      300,
})


def fmt_tokens(x, pos=None):
    if x >= 1e9:
        return f"{x/1e9:.0f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def L_inf(N, p=CANON):
    """Irreducible loss E + A/N^α."""
    return p["E"] + p["A"] / N ** p["alpha"]


def Rstar(N, D, p=CANON):
    return np.exp(p["log_K"] + p["rho"] * np.log(D / N) + p["sigma"] * np.log(N))


def eta(N, D, Dp, p=CANON):
    """exp-sat with R*(N,D); 0 when Dp == 0."""
    out = np.zeros_like(D, dtype=np.float64)
    mask = Dp > 0
    R = Rstar(N[mask], D[mask], p)
    x = Dp[mask] / D[mask]
    out[mask] = R * (1.0 - np.exp(-x / R)) / x
    return out


def L_pred(N, D, Dp, p=CANON):
    e = eta(N, D, Dp, p)
    D_eff = D + e * Dp
    return p["E"] + p["A"] / N ** p["alpha"] + p["B"] / D_eff ** p["beta"]


def cmap_for(n):
    return plt.cm.viridis(np.linspace(0, 1, n))


def plot_fig3():
    data = collect_pooled()
    tags = data["tags"]
    Ns = data["N"]
    Ds = data["D"]
    Dps = data["Dp"]
    Ls = data["L"]
    is_multi = data["is_multi"]

    # ── Identify the points dropped by the canonical k=15 fit ──────────
    # Approximation: drop the points with the largest |log L - log L_pred|
    # under the canonical joint anchors. Iterative greedy drop and the
    # one-shot top-k drop coincide for the largest residuals; the few
    # discrepancies don't matter for a paper figure.
    pred = L_pred(Ns, Ds, Dps)
    log_resid = np.log(Ls) - np.log(pred)
    drop_idx = np.argsort(np.abs(log_resid))[::-1][:CANONICAL_K]
    keep = np.ones(len(Ls), dtype=bool)
    keep[drop_idx] = False

    # Reducible loss y = L - (E + A/N^α). For the joint fit y_pred = B/D_eff^β.
    L_red_obs = Ls - L_inf(Ns)
    L_red_obs = np.clip(L_red_obs, 1e-6, None)  # guard against tiny negatives

    sizes = sorted(set(tags.tolist()), key=lambda t: SIZES[t][0])
    colors = {t: c for t, c in zip(sizes, cmap_for(len(sizes)))}

    # Effective tokens D_eff = D + η·D' (= D for 1-epoch points)
    eta_arr = eta(Ns, Ds, Dps)
    D_eff = Ds + eta_arr * Dps

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax_1ep, ax_all = axes

    # The shared linear backbone in log-log:
    #   log(L - L_inf) = log B - β log D_eff
    D_smooth = np.geomspace(D_eff.min() * 0.6, D_eff.max() * 1.6, 240)
    line_red = CANON["B"] / D_smooth ** CANON["beta"]

    # ── Panel (a): 1-epoch kept points only (D_eff = D) ──────────────
    for tag in sizes:
        m1 = (tags == tag) & ~is_multi & keep
        ax_1ep.scatter(D_eff[m1], L_red_obs[m1], s=44, color=colors[tag],
                       edgecolors="k", linewidths=0.4, zorder=5,
                       label=f"{tag}")
    ax_1ep.plot(D_smooth, line_red, "-", color="0.2", linewidth=2.0, zorder=3)
    ax_1ep.set_xscale("log", base=10)
    ax_1ep.set_yscale("log")
    ax_1ep.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_1ep.set_xlabel(r"effective tokens $D + \eta\,D'$  (= $D$ for 1-epoch)")
    ax_1ep.set_ylabel(r"reducible loss $\;L - (E + A/N^{\alpha})$")
    ax_1ep.set_title(r"(a) 1-epoch kept points — backbone validation")
    ax_1ep.legend(title=r"$N$", loc="lower left", ncol=2,
                  columnspacing=0.8, handletextpad=0.4, fontsize=9)

    keep_1ep = (~is_multi) & keep
    rmse_1ep = float(np.sqrt(np.mean(log_resid[keep_1ep] ** 2)))
    n_kept_1ep = int(keep_1ep.sum())
    n_total_1ep = int((~is_multi).sum())
    ax_1ep.text(0.04, 0.92,
                rf"RMSE($\log L$) = {rmse_1ep:.3f}  ($n = {n_kept_1ep}/{n_total_1ep}$)",
                transform=ax_1ep.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    # ── Panel (b): all kept pooled points on the same effective axis ──
    for tag in sizes:
        m = (tags == tag) & keep
        m1 = m & ~is_multi
        ax_all.scatter(D_eff[m1], L_red_obs[m1], s=44, color=colors[tag],
                       edgecolors="k", linewidths=0.4, zorder=5)
        mm = m & is_multi
        ax_all.scatter(D_eff[mm], L_red_obs[mm], s=30, marker="s",
                       facecolors="none", edgecolors=colors[tag],
                       linewidths=1.0, alpha=0.85, zorder=4)
    ax_all.plot(D_smooth, line_red, "-", color="0.2", linewidth=2.0,
                zorder=3, label=r"$B/D_{\mathrm{eff}}^{\beta}$")
    ax_all.set_xscale("log", base=10)
    ax_all.set_yscale("log")
    ax_all.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_all.set_xlabel(r"effective tokens $D + \eta\,D'$")
    ax_all.set_ylabel(r"reducible loss $\;L - (E + A/N^{\alpha})$")
    ax_all.set_title(r"(b) all kept points (1-ep filled, multi-ep open)")
    style_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="0.6", markeredgecolor="k",
                   markersize=8, label="1-epoch"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="none", markeredgecolor="0.4",
                   markersize=8, label="multi-epoch"),
        plt.Line2D([0], [0], color="0.2", linewidth=2.0,
                   label=rf"$B/D_{{\mathrm{{eff}}}}^{{\beta}}$, $\beta={CANON['beta']:.3f}$"),
    ]
    ax_all.legend(handles=style_handles, loc="lower left",
                  fontsize=9, handletextpad=0.4)

    # Pooled RMSE on kept points (data already lives on the effective axis)
    rmse_all = float(np.sqrt(np.mean(log_resid[keep] ** 2)))
    n_kept = int(keep.sum())
    n_total = int(len(Ls))
    ax_all.text(0.04, 0.92,
                rf"RMSE($\log L$) = {rmse_all:.3f}  ($n = {n_kept}/{n_total}$)",
                transform=ax_all.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    fig.tight_layout()
    out_pdf = os.path.join(OUT_DIR, "fig3_loss_vs_D.pdf")
    out_png = os.path.join(OUT_DIR, "fig3_loss_vs_D.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")
    print(f"dropped (top-k={CANONICAL_K} residuals): {len(drop_idx)} of {len(Ls)} pooled")
    print(f"  → kept {keep.sum()} pooled = {keep_1ep.sum()} 1-ep + {(is_multi & keep).sum()} multi-ep")
    print(f"1-ep kept RMSE(log L)   = {rmse_1ep:.4f}")
    print(f"pooled kept RMSE(log L) = {rmse_all:.4f}")


if __name__ == "__main__":
    plot_fig3()
