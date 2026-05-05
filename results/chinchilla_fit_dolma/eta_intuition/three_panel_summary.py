"""
Three-panel summary combining the per-N (30M) and N-dependency views.

  • Panel 1 — 30M:   y = η,        x = D/N,    color = D'/D, lines = 30M refit
  • Panel 2 — 30M:   y = η,        x = D'/D,   color = D/N,  lines = 30M refit
  • Panel 3 — multi-N at 2×: y = η·D'/D, x = D'/D, color = N (cividis),
                              lines = multi-N k=15 anchors

Pipeline / fit anchors are NOT re-run.  Panel 3 uses the canonical k=15
anchors from writeup_final.md §2.  Panels 1 and 2 use the writeup §1.3
30M-only 1-epoch anchors and refit (log K_eff, ρ) on the multi-epoch 30M
points (same procedure as per_N_analysis/test_30M/empirical_eta_30m.py).

Output: eta_intuition/three_panel_summary.pdf
"""

import glob
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import OVERFIT_EXCLUDE, SIZES, TTP_RATIO, extract_multi_epoch  # noqa: E402
from fit_eta import (collect_multi_epoch_all_sizes, load,  # noqa: E402
                     per_point_eta, per_point_eta_diff)
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402


# ── 30M-only 1-epoch anchors (writeup.md §1.3, k=3, δ=0.1) ───────────
SIZE_30M = "30m"
E_30M, B_30M, BETA_30M = 3.010, 47_022.0, 0.4863

# ── Multi-N joint η anchors (writeup_final.md §2, one-shot k=15) ─────
LOG_K, RHO, SIGMA = 10.32, -0.270, -0.388
B_JOINT, BETA_JOINT = 16_539.0, 0.436

# Discrete grids present in the data
DoN_VALUES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
DpD_VALUES = (1.0, 3.0, 7.0, 15.0, 31.0, 63.0)
SCALE_PANEL3 = 2.0
SCALE_TOL = 0.05

# Curve sweep ranges
DoN_MIN, DoN_MAX = 0.5, 500.0
DpD_MIN, DpD_MAX = 0.3, 200.0
NPTS = 200

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── matplotlib look-and-feel ────────────────────────────────────────
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
    'legend.fontsize':   9,
    'lines.linewidth':   1.8,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '-',
    'savefig.bbox':      'tight',
    'savefig.dpi':       300,
})


# Color scales
DpD_NORM = LogNorm(vmin=min(DpD_VALUES), vmax=max(DpD_VALUES))
DpD_CMAP = plt.cm.plasma
DoN_NORM = LogNorm(vmin=min(DoN_VALUES), vmax=max(DoN_VALUES))
DoN_CMAP = plt.cm.viridis


def Rstar_30M(D, N, log_K_eff, rho):
    return np.exp(log_K_eff + rho * np.log(D / N))


def Rstar_KN(D, N):
    return np.exp(LOG_K + RHO * np.log(D / N) + SIGMA * np.log(N))


# ── 30M empirical + refit (mirrors test_30M/empirical_eta_30m.py) ───
def load_30m_and_refit():
    N, datasets = load(SIZE_30M)
    s, D, ep, Dp, L, _ = extract_multi_epoch(
        datasets, N, scale_min=0.0,
        exclude_overfit=OVERFIT_EXCLUDE.get(SIZE_30M, set()))
    D, Dp, L = np.asarray(D), np.asarray(Dp), np.asarray(L)
    eta = per_point_eta(D, Dp, L, E_30M, B_30M, BETA_30M)
    DoN, DpD = D / N, Dp / D

    # Refit (log K_eff, ρ) with frozen (E, B, β) — same as test_30M
    D_t = torch.as_tensor(D, dtype=torch.float64)
    Dp_t = torch.as_tensor(Dp, dtype=torch.float64)
    log_L_obs = torch.log(torch.as_tensor(L, dtype=torch.float64))
    log_D = torch.log(D_t)
    log_DoN = log_D - math.log(N)
    x = Dp_t / D_t
    log_E, log_B = math.log(E_30M), math.log(B_30M)

    def fwd(p):
        log_R = p["log_K_eff"] + p["rho"] * log_DoN
        R = torch.exp(log_R)
        log_D_eff = log_D + torch.log1p(R * (1.0 - torch.exp(-x / R)))
        return logsumexp_stable(
            torch.stack([torch.full_like(log_D_eff, log_E),
                         log_B - BETA_30M * log_D_eff]), dim=0)

    init = expand_grid({"log_K_eff": [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
                        "rho":       [-1.0, -0.5, -0.2, 0.0, 0.2]})
    fit = fit_lse(fwd, log_L_obs, init, delta=0.1, verbose=False)
    return N, DoN, DpD, eta, fit


# ── Panels ──────────────────────────────────────────────────────────
def panel_30m_vs_DoverN(ax, N, DoN, DpD, eta, fit):
    log_K_eff, rho = fit["params"]["log_K_eff"], fit["params"]["rho"]
    DoN_grid = np.geomspace(DoN_MIN, DoN_MAX, NPTS)
    R = Rstar_30M(DoN_grid * N, N, log_K_eff, rho)
    for v in DpD_VALUES:
        y = R * (1.0 - np.exp(-v / R)) / v
        ax.plot(DoN_grid, y, '-', color=DpD_CMAP(DpD_NORM(v)),
                linewidth=2.0, alpha=0.95, zorder=2)

    valid = ~np.isnan(eta) & (eta >= 0) & (DoN > 1.0)
    for v in DpD_VALUES:
        m = valid & (np.abs(DpD - v) <= 0.01)
        if not m.any():
            continue
        ax.scatter(DoN[m], eta[m], s=44, color=DpD_CMAP(DpD_NORM(v)),
                   edgecolors='k', linewidths=0.4, alpha=0.95, zorder=3,
                   label=rf"$D'/D = {v:g}$")
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"data scale $D/N$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(rf"(a) 30M: $\eta$ vs $D/N$")
    ax.legend(title=r"$D'/D$", loc="lower left", ncol=2,
              columnspacing=0.8, handletextpad=0.3)


def panel_30m_vs_DpoverD(ax, N, DoN, DpD, eta, fit):
    log_K_eff, rho = fit["params"]["log_K_eff"], fit["params"]["rho"]
    DpD_grid = np.geomspace(DpD_MIN, DpD_MAX, NPTS)
    for v in DoN_VALUES:
        R = Rstar_30M(v * N, N, log_K_eff, rho)
        y = R * (1.0 - np.exp(-DpD_grid / R)) / DpD_grid
        ax.plot(DpD_grid, y, '-', color=DoN_CMAP(DoN_NORM(v)),
                linewidth=2.0, alpha=0.95, zorder=2,
                label=rf"$D/N = {v:g}$")

    valid = ~np.isnan(eta) & (eta >= 0) & (DoN > 1.0)
    for v in DoN_VALUES:
        m = valid & (np.abs(DoN - v) <= 0.01 * v)
        if not m.any():
            continue
        ax.scatter(DpD[m], eta[m], s=44, color=DoN_CMAP(DoN_NORM(v)),
                   edgecolors='k', linewidths=0.4, alpha=0.95, zorder=3)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$D'/D$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(rf"(b) 30M: $\eta$ vs $D'/D$")
    ax.legend(title=r"$D/N$", loc="lower left", ncol=2,
              columnspacing=0.8, handletextpad=0.3)


def panel_multiN_at_2x(ax, scale=SCALE_PANEL3):
    """y = η·D'/D vs x = D'/D at scale 2× across sizes (cividis)."""
    tags, Ns, scales, Ds, _eps, Dps, Ls, L_1eps = \
        collect_multi_epoch_all_sizes(0.0)
    eta_pp = per_point_eta_diff(Ds, Dps, Ls, L_1eps, B_JOINT, BETA_JOINT)
    valid = (~np.isnan(eta_pp) & (eta_pp >= 0) & (eta_pp <= 5)
             & (np.abs(np.asarray(scales) - scale) <= SCALE_TOL * scale))

    sizes_plot = sorted(["14m", "30m", "60m", "190m", "370m"],
                        key=lambda t: SIZES[t][0])
    cmap = plt.cm.cividis(np.linspace(0.05, 0.95, len(sizes_plot)))
    colors = {t: c for t, c in zip(sizes_plot, cmap)}

    DpD_grid = np.geomspace(0.5, 256, 300)
    for tag in sizes_plot:
        N = SIZES[tag][0]
        D = scale * TTP_RATIO * N
        R = Rstar_KN(D, N)
        y = R * (1.0 - np.exp(-DpD_grid / R))   # = η · D'/D
        ax.plot(DpD_grid, y, '-', color=colors[tag], linewidth=2.0,
                label=f"{tag}  $R^*$={R:.1f}")
        ax.axhline(R, color=colors[tag], linestyle=':', linewidth=1.0,
                   alpha=0.5)

    for tag in sizes_plot:
        m = (tags == tag) & valid
        if not m.any():
            continue
        DpD_emp = Dps[m] / Ds[m]
        ax.scatter(DpD_emp, eta_pp[m] * DpD_emp, s=40, color=colors[tag],
                   edgecolors='k', linewidths=0.4, alpha=0.9, zorder=3)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$D'/D$")
    ax.set_ylabel(r"$\eta \cdot D'/D$")
    ax.set_title(rf"(c) multi-$N$ at ${scale}\times$:  "
                 rf"dotted = $R^*$ asymptote")
    ax.legend(loc='lower right')


if __name__ == "__main__":
    N, DoN, DpD, eta, fit = load_30m_and_refit()
    print(f"30M refit: log K_eff={fit['params']['log_K_eff']:.3f}, "
          f"ρ={fit['params']['rho']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
    panel_30m_vs_DoverN(axes[0], N, DoN, DpD, eta, fit)
    panel_30m_vs_DpoverD(axes[1], N, DoN, DpD, eta, fit)
    panel_multiN_at_2x(axes[2])

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "three_panel_summary.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")
