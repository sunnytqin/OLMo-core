"""
30M-only empirical η scatter, with the multi-N joint η model overlaid.

Fixes N = 30M and uses the 30M-only 1-epoch Chinchilla anchors from
chinchilla_fit_dolma/writeup.md §1.3 to *solve* η empirically:

    L = E + B / D^β,    E = 3.010,  B = 47_022,  β = 0.4863
    (k=3 cut, δ = 0.1)

For each multi-epoch 30M data point we solve η directly from

    L = E + B / (D + η · D')^β   ⇒   η = ((B/(L − E))^{1/β} − D) / D'

(per_point_eta in fit_eta.py — the direct solve, not ΔL).

Overlay curves use the canonical multi-N one-shot k=15 η anchors from
writeup_final.md §2:

    R*(D, N) = exp(log K + ρ log(D/N) + σ log N)
    log K = 10.32,  ρ = −0.270,  σ = −0.388
    η · D'/D = R*(1 − exp(−x/R*)),     x = D'/D

(Note R* does not depend on B or β, so the curve is well-defined even
though the multi-N joint fit used different B, β than the 30M-only fit.
Disagreement = the multi-N model not transferring to a 30M-specific β.)

Two scatter plots, both with y = η · D'/D :
  • plot 1: x = D/N   (data scale),  colored by D'/D
  • plot 2: x = D'/D  (epoch ratio), colored by D/N

Output:
  test_30M/empirical_eta_30m_vs_DoverN.pdf
  test_30M/empirical_eta_30m_vs_DpoverD.pdf
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data import OVERFIT_EXCLUDE, SIZES, extract_multi_epoch  # noqa: E402
from fit_eta import load, per_point_eta  # noqa: E402
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402


# ── 30M-only 1-epoch anchors (writeup.md §1.3, k=3, δ=0.1) ───────────
SIZE_TAG = "30m"
E_FIT = 3.010
B_FIT = 47_022.0
BETA_FIT = 0.4863

# ── Multi-N η anchors (writeup_final.md §2, one-shot k=15) ───────────
LOG_K = 10.32
RHO = -0.270
SIGMA = -0.388

# Discrete grids: includes D/N=0.5 below the data range (curve only).
DoN_VALUES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
DpD_VALUES = (1.0, 3.0, 7.0, 15.0, 31.0, 63.0)

# Curve sweep ranges
DoN_MIN, DoN_MAX, DoN_NPTS = 0.5, 500.0, 200
DpD_MIN, DpD_MAX, DpD_NPTS = 0.3, 200.0, 200

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── matplotlib look-and-feel matching paper_figures.py ───────────────
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
    'lines.linewidth':   1.8,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '-',
    'savefig.bbox':      'tight',
    'savefig.dpi':       300,
})


# Color scales for the two plots
DpD_NORM = LogNorm(vmin=min(DpD_VALUES), vmax=max(DpD_VALUES))
DpD_CMAP = plt.cm.plasma
DoN_NORM = LogNorm(vmin=min(DoN_VALUES), vmax=max(DoN_VALUES))
DoN_CMAP = plt.cm.viridis


def Rstar_KN(D, N, log_K=LOG_K, rho=RHO, sigma=SIGMA):
    """Multi-N anchor R* (writeup_final.md §2 k=15)."""
    return np.exp(log_K + rho * np.log(D / N) + sigma * np.log(N))


def Rstar_30M(D, N, log_K_eff, rho):
    """30M-only R*: with N fixed, σ·log N is absorbed into log K_eff."""
    return np.exp(log_K_eff + rho * np.log(D / N))


# ── 30M-only refit: same form, same fitter, 30M anchors ─────────────
def fit_30m_eta(D_np, Dp_np, L_np, N):
    """Refit (log K_eff, ρ) on the 30M multi-epoch points using the same
    LSE + Huber + L-BFGS pipeline as the writeup, with (E, B, β) frozen
    at the 30M-only 1-epoch anchors.  At fixed N, R* = K_eff (D/N)^ρ —
    σ is not separately identifiable, so the multi-N (log K, σ) collapse
    into a single log K_eff.
    """
    D = torch.as_tensor(D_np, dtype=torch.float64)
    Dp = torch.as_tensor(Dp_np, dtype=torch.float64)
    log_L_obs = torch.log(torch.as_tensor(L_np, dtype=torch.float64))
    log_D = torch.log(D)
    log_DoN = log_D - math.log(N)
    x = Dp / D
    log_E, log_B, beta = math.log(E_FIT), math.log(B_FIT), BETA_FIT

    def forward(p):
        log_R = p["log_K_eff"] + p["rho"] * log_DoN
        R = torch.exp(log_R)
        # D_eff = D · (1 + R · (1 − e^{−x/R}))     (since D'/D = x)
        log_D_eff = log_D + torch.log1p(R * (1.0 - torch.exp(-x / R)))
        # log L = logsumexp(log E, log B − β·log D_eff)
        return logsumexp_stable(
            torch.stack([torch.full_like(log_D_eff, log_E),
                         log_B - beta * log_D_eff]),
            dim=0,
        )

    # Init grid: anchor near the multi-N transferred values
    # (log K_eff ≈ log K + σ·log N = 10.32 − 0.388·log(3e7) ≈ 3.6, ρ ≈ −0.27).
    init = expand_grid({
        "log_K_eff": [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
        "rho":       [-1.0, -0.5, -0.2, 0.0, 0.2],
    })
    fit = fit_lse(forward, log_L_obs, init, delta=0.1, verbose=False)
    return fit


def _load_30m_empirical():
    """Load 30M multi-epoch data and solve η directly from the fitted
    1-epoch (E, B, β)."""
    N, datasets = load(SIZE_TAG)
    s, D, ep, Dp, L, L_1ep = extract_multi_epoch(
        datasets, N, scale_min=0.0,
        exclude_overfit=OVERFIT_EXCLUDE.get(SIZE_TAG, set()))
    D = np.asarray(D)
    Dp = np.asarray(Dp)
    L = np.asarray(L)
    eta = per_point_eta(D, Dp, L, E_FIT, B_FIT, BETA_FIT)
    DoN = D / N
    DpD = Dp / D
    return N, D, Dp, L, eta, DoN, DpD


def plot_vs_DoverN(N, DoN, DpD, eta, fit30):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # ── 30M-refit η curves: one per D'/D ─────────────────────────────
    DoN_grid = np.geomspace(DoN_MIN, DoN_MAX, DoN_NPTS)
    D_grid = DoN_grid * N
    R_30 = Rstar_30M(D_grid, N, fit30["params"]["log_K_eff"],
                     fit30["params"]["rho"])
    for v in DpD_VALUES:
        y_30 = R_30 * (1.0 - np.exp(-v / R_30)) / v   # = η at fixed D'/D = v
        ax.plot(DoN_grid, y_30, '-', color=DpD_CMAP(DpD_NORM(v)),
                linewidth=2.0, alpha=0.95, zorder=2)

    # ── empirical scatter (discrete D'/D bins) ───────────────────────
    valid = ~np.isnan(eta) & (eta >= 0) & (DoN > 1.0)
    n_kept = 0
    for v in DpD_VALUES:
        m = valid & (np.abs(DpD - v) <= 0.01)
        if not m.any():
            continue
        ax.scatter(DoN[m], eta[m], s=56, color=DpD_CMAP(DpD_NORM(v)),
                   edgecolors='k', linewidths=0.4, alpha=0.95, zorder=3,
                   label=rf"$D'/D = {v:g}$")
        n_kept += int(m.sum())
    print(f"vs D/N : kept {n_kept} / {valid.sum()} valid pts "
          f"({len(eta)} total, {(~valid).sum()} unphysical)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.set_xlabel(r"data scale $D/N$  (tokens-per-param)")
    ax.set_ylabel(r"$\eta$  (effective-token multiplier)")
    ax.set_title(r"30M $\eta$ vs $D/N$   (lines: 30M refit)")
    ax.legend(title=r"$D'/D$", loc="best", ncol=2,
              columnspacing=1.0, handletextpad=0.4)
    ax.text(0.04, 0.04,
            rf"30M-refit: $\log K_\mathrm{{eff}}={fit30['params']['log_K_eff']:.2f}$, "
            rf"$\rho={fit30['params']['rho']:.2f}$    "
            rf"RMSE$_{{\log L}}={fit30['rmse_logL']:.3f}$",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "empirical_eta_30m_vs_DoverN.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"   → {out}")


def plot_vs_DpoverD(N, DoN, DpD, eta, fit30):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # ── 30M-refit curves: one per D/N, x = D'/D ──────────────────────
    DpD_grid = np.geomspace(DpD_MIN, DpD_MAX, DpD_NPTS)
    for v in DoN_VALUES:
        D = v * N
        R_30 = Rstar_30M(D, N, fit30["params"]["log_K_eff"],
                         fit30["params"]["rho"])
        y_30 = R_30 * (1.0 - np.exp(-DpD_grid / R_30)) / DpD_grid   # = η
        ax.plot(DpD_grid, y_30, '-', color=DoN_CMAP(DoN_NORM(v)),
                linewidth=2.0, alpha=0.95, zorder=2,
                label=rf"$D/N = {v:g}$")

    # ── empirical scatter (discrete D/N bins; curves carry the legend) ─
    valid = ~np.isnan(eta) & (eta >= 0) & (DoN > 1.0)
    n_kept = 0
    for v in DoN_VALUES:
        m = valid & (np.abs(DoN - v) <= 0.01 * v)
        if not m.any():
            continue
        ax.scatter(DpD[m], eta[m], s=56, color=DoN_CMAP(DoN_NORM(v)),
                   edgecolors='k', linewidths=0.4, alpha=0.95, zorder=3)
        n_kept += int(m.sum())
    print(f"vs D'/D: kept {n_kept} / {valid.sum()} valid pts "
          f"({len(eta)} total, {(~valid).sum()} unphysical)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, zorder=1)
    ax.set_xlabel(r"$D'/D$  (epoch ratio = epochs $-1$)")
    ax.set_ylabel(r"$\eta$  (effective-token multiplier)")
    ax.set_title(r"30M $\eta$ vs $D'/D$   (lines: 30M refit)")
    ax.legend(title=r"$D/N$", loc="best", ncol=2,
              columnspacing=1.0, handletextpad=0.4)
    ax.text(0.04, 0.04,
            rf"30M-refit: $\log K_\mathrm{{eff}}={fit30['params']['log_K_eff']:.2f}$, "
            rf"$\rho={fit30['params']['rho']:.2f}$    "
            rf"RMSE$_{{\log L}}={fit30['rmse_logL']:.3f}$",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "empirical_eta_30m_vs_DpoverD.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"   → {out}")


if __name__ == "__main__":
    N, D, Dp, L, eta, DoN, DpD = _load_30m_empirical()
    print(f"30M N = {N:.2g}, multi-epoch points = {len(D)}")
    print(f"  unphysical (L < E or solve failed): {int(np.isnan(eta).sum())}")

    fit30 = fit_30m_eta(D, Dp, L, N)
    p = fit30["params"]
    print(f"30M-refit (Huber+L-BFGS, δ=0.1): log K_eff={p['log_K_eff']:.4f}, "
          f"ρ={p['rho']:.4f}")
    print(f"   loss={fit30['loss']:.4f}, RMSE(log L)={fit30['rmse_logL']:.4f}, "
          f"R²={fit30['r2_logL']:.4f}")
    print(f"   multi-N transfer at N=30M: log K_eff="
          f"{LOG_K + SIGMA*math.log(N):.4f}, ρ={RHO:.4f}")

    plot_vs_DoverN(N, DoN, DpD, eta, fit30)
    plot_vs_DpoverD(N, DoN, DpD, eta, fit30)
