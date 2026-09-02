"""
Figure 3 for the NeurIPS submission (paper section 4.4):
  Reducible validation loss  L_red = L - (E + A/N^α)  vs tokens on log-log
  axes. Under the triple-joint fit L = E + A/N^α + B/D_eff^β, the reducible
  loss B/D_eff^β is a single straight line of slope -β across all N.

  Panel (a): raw tokens D. 1-epoch points sit on the classic scaling-law
             line; repetition (squares) and paraphrase (triangles) points
             fall BELOW it — at the same fresh D they trained on more
             effective tokens, so their reducible loss is lower.
  Panel (b): effective tokens D_eff = D + η_src·D'. All three sources
             collapse back onto the same line, validating the η correction.

  Points dropped by the canonical k=15 residual trim (approximated as the
  top-k pooled-residual points against the headline joint-fit prediction)
  are excluded.

Saves fig3_loss_vs_D.pdf and PNG in this directory.
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
from fit_joint_triple import (  # noqa: E402
    SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA, collect_pooled_triple,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# ── Headline triple-joint fit (final results) ──────────────────────────
#   backbone  L = E + A/N^α + B/D_eff^β
#   per-strategy saturation  R*(N, D) = exp(logK + ρ·log(D/N) + σ·log N),
#                            η = R*(1 - e^{-x/R*}) / x,   x = D'/D
CANON = dict(
    E=1.35, A=205.0, B=16_597.0, alpha=0.283, beta=0.435,
    log_K_rep=10.93,  rho_rep=-0.42,  sigma_rep=-0.41,
    log_K_para=30.50, rho_para=-1.52, sigma_para=-1.30,
)
CANONICAL_K = 15  # drop top-15 worst-residual points before plotting


# ── Palatino (URW P052 clone) — match paper_figures.py styling ──────
for _f in glob.glob("/usr/share/fonts/urw-base35/P052-*.otf"):
    font_manager.fontManager.addfont(_f)

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["P052", "Palatino", "TeX Gyre Pagella", "serif"],
    "mathtext.fontset": "cm",
    "font.size":        20,
    "axes.titlesize":   22,
    "axes.labelsize":   22,
    "xtick.labelsize":  18,
    "ytick.labelsize":  18,
    "legend.fontsize":  16,
    "figure.titlesize": 24,
    "lines.linewidth":  2.2,
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


def Rstar(N, D, log_K, rho, sigma):
    return np.exp(log_K + rho * np.log(D / N) + sigma * np.log(N))


def _eta_form(R, x):
    """exp-sat: η = R*(1 - e^{-x/R}) / x for x > 0."""
    return R * (1.0 - np.exp(-x / R)) / x


def eta_per_source(N, D, Dp, source, p=CANON):
    """Strategy-aware η: 0 for 1-epoch, η_rep for repeat, η_para for para."""
    out = np.zeros_like(D, dtype=np.float64)
    is_rep  = (source == SOURCE_REPEAT) & (Dp > 0)
    is_para = (source == SOURCE_PARA)   & (Dp > 0)
    if is_rep.any():
        R = Rstar(N[is_rep], D[is_rep],
                  p["log_K_rep"], p["rho_rep"], p["sigma_rep"])
        out[is_rep] = _eta_form(R, Dp[is_rep] / D[is_rep])
    if is_para.any():
        R = Rstar(N[is_para], D[is_para],
                  p["log_K_para"], p["rho_para"], p["sigma_para"])
        out[is_para] = _eta_form(R, Dp[is_para] / D[is_para])
    return out


def L_pred(N, D, Dp, source, p=CANON):
    D_eff = D + eta_per_source(N, D, Dp, source, p) * Dp
    return p["E"] + p["A"] / N ** p["alpha"] + p["B"] / D_eff ** p["beta"]


def cmap_for(n):
    return plt.cm.viridis(np.linspace(0, 1, n))


def plot_fig3():
    data = collect_pooled_triple()
    tags = data["tags"]
    Ns = data["N"]
    Ds = data["D"]
    Dps = data["Dp"]
    Ls = data["L"]
    src = data["source"]

    is_1ep  = (src == SOURCE_NONE)
    is_rep  = (src == SOURCE_REPEAT)
    is_para = (src == SOURCE_PARA)

    # ── Identify the points dropped by the canonical k=15 fit ──────────
    # Approximation: drop the points with the largest |log L - log L_pred|
    # under the headline anchors. Iterative greedy drop and the one-shot
    # top-k drop coincide for the largest residuals.
    pred = L_pred(Ns, Ds, Dps, src)
    log_resid = np.log(Ls) - np.log(pred)
    drop_idx = np.argsort(np.abs(log_resid))[::-1][:CANONICAL_K]
    keep = np.ones(len(Ls), dtype=bool)
    keep[drop_idx] = False

    # Reducible loss y = L - (E + A/N^α); backbone prediction y = B/D_eff^β.
    L_red_obs = np.clip(Ls - L_inf(Ns), 1e-6, None)

    # Effective tokens D_eff = D + η_src·D' (= D for 1-epoch points)
    eta_arr = eta_per_source(Ns, Ds, Dps, src)
    D_eff = Ds + eta_arr * Dps

    sizes = sorted(set(tags.tolist()), key=lambda t: SIZES[t][0])
    colors = {t: c for t, c in zip(sizes, cmap_for(len(sizes)))}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    ax_raw, ax_eff = axes

    # Shared linear backbone: log(L - L_inf) = log B - β log D
    x_lo = min(Ds.min(), D_eff.min()) * 0.6
    x_hi = max(Ds.max(), D_eff.max()) * 1.6
    D_smooth = np.geomspace(x_lo, x_hi, 240)
    line_red = CANON["B"] / D_smooth ** CANON["beta"]

    y_label = "Data-Reducible Loss\n" + r"$L - (E + A/N^{\alpha})$"

    def scatter_panel(ax, x_vals):
        for tag in sizes:
            m = (tags == tag) & keep
            m1 = m & is_1ep
            ax.scatter(x_vals[m1], L_red_obs[m1], s=64, color=colors[tag],
                       edgecolors="k", linewidths=0.4, zorder=5, label=tag)
            mr = m & is_rep
            ax.scatter(x_vals[mr], L_red_obs[mr], s=30, marker="s",
                       facecolors="none", edgecolors=colors[tag],
                       linewidths=1.0, alpha=0.85, zorder=4)
            mp = m & is_para
            ax.scatter(x_vals[mp], L_red_obs[mp], s=42, marker="^",
                       facecolors="none", edgecolors=colors[tag],
                       linewidths=1.1, alpha=0.9, zorder=4)
        ax.plot(D_smooth, line_red, "-", color="0.2", linewidth=2.0, zorder=3,
                label="Classic Scaling Law Fit")
        ax.set_xscale("log", base=10)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))

    # ── Panel (a): raw tokens D — extra-data points fall below the line ─
    scatter_panel(ax_raw, Ds)
    ax_raw.set_xlabel(r"Tokens $D$")
    ax_raw.set_ylabel(y_label)

    # ── Panel (b): effective tokens D_eff — all sources collapse ───────
    scatter_panel(ax_eff, D_eff)
    ax_eff.set_xlabel(r"Effective Tokens $D_{\mathrm{eff}} = D + \eta_{\mathrm{strat}}\,D'$")

    # ── Legends: left panel gets the shape/style legend, right the N ───
    style_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="0.6", markeredgecolor="k",
                   markersize=13, label="1-Epoch"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="none", markeredgecolor="0.4",
                   markersize=8, label="Repetition"),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor="none", markeredgecolor="0.4",
                   markersize=8, label="Paraphrase"),
    ]
    shape_leg = ax_raw.legend(handles=style_handles, loc="lower left",
                              fontsize=16, handletextpad=0.5, frameon=False)
    ax_raw.add_artist(shape_leg)
    fit_handle = [plt.Line2D([0], [0], color="0.2", linewidth=2.0,
                             label="Classic Scaling Law Fit")]
    ax_raw.legend(handles=fit_handle, loc="upper right",
                  fontsize=16, handletextpad=0.5, frameon=False)

    size_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[t],
                   markeredgecolor="k", markersize=10, label=t)
        for t in sizes
    ]
    ax_eff.legend(handles=size_handles, title=r"$N$", loc="lower left",
                  ncol=2, columnspacing=0.8, handletextpad=0.4, fontsize=14,
                  frameon=False)

    fig.tight_layout()
    out_pdf = os.path.join(OUT_DIR, "fig3_loss_vs_D.pdf")
    out_png = os.path.join(OUT_DIR, "fig3_loss_vs_D.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")
    print(f"dropped top-k={CANONICAL_K} residuals; kept {keep.sum()}/{len(Ls)} "
          f"= {(is_1ep & keep).sum()} 1-ep + {(is_rep & keep).sum()} rep + "
          f"{(is_para & keep).sum()} para")


if __name__ == "__main__":
    plot_fig3()
