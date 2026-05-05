"""
Single-size η pipeline for 14M (mirror of test_30M/empirical_eta_30m.py).

Steps (all done in this script — no hardcoded anchors):
  1. 1-epoch Chinchilla fit at min_scale = 0.5, δ = 0.1, via the shared
     LSE + Huber + L-BFGS pipeline (matches writeup §1.3 procedure).
        L = E + B / D^β
  2. Solve empirical η per multi-epoch point from
        L = E + B / (D + η · D')^β
     (per_point_eta in fit_eta.py — direct solve, not ΔL).
  3. Refit (log K_eff, ρ) on the multi-epoch points using the same
     fit_lse machinery, with (E, B, β) frozen.  At fixed N, σ collapses
     into log K_eff so the η model has 2 free params.
  4. Two scatter plots, both with y = η · D'/D :
       • plot 1: x = D/N   (data scale),  curves & color by D'/D
       • plot 2: x = D'/D  (epoch ratio), curves & color by D/N

Output:
  test_14M/empirical_eta_14m_vs_DoverN.pdf
  test_14M/empirical_eta_14m_vs_DpoverD.pdf
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

from data import OVERFIT_EXCLUDE, extract_1epoch, extract_multi_epoch  # noqa: E402
from fit_eta import load, per_point_eta  # noqa: E402
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402


# ── Per-size knobs ──────────────────────────────────────────────────
SIZE_TAG = "60m"
MIN_SCALE_1EP = 0.0    # use all 1-epoch points (no scale cut)
DELTA = 0.1            # Huber δ for both fits
BETA_FIXED = 0.436     # fixed at multi-N joint k=15 β
# Multi-N joint anchors for E_eff(N) = E0 + A/N^α  (writeup_final.md §2 k=15)
MULTI_N_E0    = 0.050
MULTI_N_A     = 31.5
MULTI_N_ALPHA = 0.137

# Discrete grids: includes D/N=0.5 below the data range (curve only).
DoN_VALUES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
DpD_VALUES = (1.0, 3.0, 7.0, 15.0, 31.0, 63.0)

# Curve sweep ranges
DoN_MIN, DoN_MAX, DoN_NPTS = 0.5, 500.0, 200
DpD_MIN, DpD_MAX, DpD_NPTS = 0.3, 200.0, 200

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── matplotlib ──────────────────────────────────────────────────────
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


def Rstar_eff(D, N, log_K_eff, rho):
    """Single-N R*: σ·log N is absorbed into log K_eff at fixed N."""
    return np.exp(log_K_eff + rho * np.log(D / N))


# ──────────────────────────────────────────────────────────────────
# Step 1 — 1-epoch fit (Huber + L-BFGS, LSE form)
# ──────────────────────────────────────────────────────────────────

def fit_1ep(D_np, L_np, E_fixed, delta=DELTA, beta_fixed=BETA_FIXED):
    """1-epoch fit with E_eff(N) and β frozen from multi-N joint formula;
    only B is optimized."""
    log_D = torch.tensor(np.log(D_np), dtype=torch.float64)
    log_L = torch.tensor(np.log(L_np), dtype=torch.float64)
    e_t = torch.tensor(math.log(E_fixed), dtype=torch.float64)
    beta_t = torch.tensor(beta_fixed, dtype=torch.float64)

    def forward(p):
        terms = torch.stack([e_t.expand_as(log_D),
                             p["b"] - beta_t * log_D], dim=0)
        return logsumexp_stable(terms, dim=0)

    grid = expand_grid({"b": [2.0, 5.0, 8.0, 11.0, 14.0]})
    res = fit_lse(forward, log_L, grid, delta=delta, verbose=False)
    p = res["params"]
    return dict(E=E_fixed,
                B=float(np.exp(p["b"])),
                beta=beta_fixed,
                rmse_logL=res["rmse_logL"],
                r2_logL=res["r2_logL"])


# ──────────────────────────────────────────────────────────────────
# Step 3 — refit (log K_eff, ρ) on multi-epoch with frozen (E, B, β)
# ──────────────────────────────────────────────────────────────────

def fit_eta_single(D_np, Dp_np, L_np, N, E, B, beta, delta=DELTA):
    D = torch.as_tensor(D_np, dtype=torch.float64)
    Dp = torch.as_tensor(Dp_np, dtype=torch.float64)
    log_L_obs = torch.log(torch.as_tensor(L_np, dtype=torch.float64))
    log_D = torch.log(D)
    log_DoN = log_D - math.log(N)
    x = Dp / D
    log_E, log_B = math.log(E), math.log(B)

    def forward(p):
        log_R = p["log_K_eff"] + p["rho"] * log_DoN
        R = torch.exp(log_R)
        # D_eff = D · (1 + R · (1 − e^{−x/R}))     (since D'/D = x)
        log_D_eff = log_D + torch.log1p(R * (1.0 - torch.exp(-x / R)))
        return logsumexp_stable(
            torch.stack([torch.full_like(log_D_eff, log_E),
                         log_B - beta * log_D_eff]),
            dim=0,
        )

    init = expand_grid({
        "log_K_eff": [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
        "rho":       [-1.0, -0.5, -0.2, 0.0, 0.2],
    })
    return fit_lse(forward, log_L_obs, init, delta=delta, verbose=False)


# ──────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────

def plot_vs_DoverN(N, DoN, DpD, eta, fit30, fit_1):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    DoN_grid = np.geomspace(DoN_MIN, DoN_MAX, DoN_NPTS)
    D_grid = DoN_grid * N
    R_30 = Rstar_eff(D_grid, N,
                     fit30["params"]["log_K_eff"], fit30["params"]["rho"])
    for v in DpD_VALUES:
        y_30 = R_30 * (1.0 - np.exp(-v / R_30))
        ax.plot(DoN_grid, y_30, '-', color=DpD_CMAP(DpD_NORM(v)),
                linewidth=2.0, alpha=0.95, zorder=2)

    valid = ~np.isnan(eta) & (eta >= 0)
    n_kept = 0
    for v in DpD_VALUES:
        m = valid & (np.abs(DpD - v) <= 0.01)
        if not m.any():
            continue
        ax.scatter(DoN[m], eta[m] * DpD[m], s=56, color=DpD_CMAP(DpD_NORM(v)),
                   edgecolors='k', linewidths=0.4, alpha=0.95, zorder=3,
                   label=rf"$D'/D = {v:g}$")
        n_kept += int(m.sum())
    print(f"vs D/N : kept {n_kept} / {valid.sum()} valid pts "
          f"({len(eta)} total, {(~valid).sum()} unphysical)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"data scale $D/N$  (tokens-per-param)")
    ax.set_ylabel(r"$\eta \cdot D'/D$  (extra fresh-equivalent tokens / $D$)")
    ax.set_title(rf"{SIZE_TAG.upper()} $\eta$ vs $D/N$   (lines: {SIZE_TAG.upper()} refit)")
    ax.legend(title=r"$D'/D$", loc="best", ncol=2,
              columnspacing=1.0, handletextpad=0.4)
    p = fit30["params"]
    ax.text(0.04, 0.04,
            rf"1-ep fit (min {MIN_SCALE_1EP}×): "
            rf"$E={fit_1['E']:.3f}$, $B={fit_1['B']:.0f}$, $\beta={fit_1['beta']:.3f}$" + "\n"
            rf"η refit:  $\log K_\mathrm{{eff}}={p['log_K_eff']:.2f}$, "
            rf"$\rho={p['rho']:.2f}$    RMSE$_{{\log L}}={fit30['rmse_logL']:.3f}$",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"empirical_eta_{SIZE_TAG}_vs_DoverN.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"   → {out}")


def plot_vs_DpoverD(N, DoN, DpD, eta, fit30, fit_1):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    DpD_grid = np.geomspace(DpD_MIN, DpD_MAX, DpD_NPTS)
    for v in DoN_VALUES:
        D = v * N
        R_30 = Rstar_eff(D, N,
                         fit30["params"]["log_K_eff"], fit30["params"]["rho"])
        y_30 = R_30 * (1.0 - np.exp(-DpD_grid / R_30))
        ax.plot(DpD_grid, y_30, '-', color=DoN_CMAP(DoN_NORM(v)),
                linewidth=2.0, alpha=0.95, zorder=2,
                label=rf"$D/N = {v:g}$")

    valid = ~np.isnan(eta) & (eta >= 0)
    n_kept = 0
    for v in DoN_VALUES:
        m = valid & (np.abs(DoN - v) <= 0.01 * v)
        if not m.any():
            continue
        ax.scatter(DpD[m], eta[m] * DpD[m], s=56, color=DoN_CMAP(DoN_NORM(v)),
                   edgecolors='k', linewidths=0.4, alpha=0.95, zorder=3)
        n_kept += int(m.sum())
    print(f"vs D'/D: kept {n_kept} / {valid.sum()} valid pts "
          f"({len(eta)} total, {(~valid).sum()} unphysical)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$D'/D$  (epoch ratio = epochs $-1$)")
    ax.set_ylabel(r"$\eta \cdot D'/D$  (extra fresh-equivalent tokens / $D$)")
    ax.set_title(rf"{SIZE_TAG.upper()} $\eta$ vs $D'/D$   (lines: {SIZE_TAG.upper()} refit)")
    ax.legend(title=r"$D/N$", loc="best", ncol=2,
              columnspacing=1.0, handletextpad=0.4)
    p = fit30["params"]
    ax.text(0.04, 0.04,
            rf"1-ep fit (min {MIN_SCALE_1EP}×): "
            rf"$E={fit_1['E']:.3f}$, $B={fit_1['B']:.0f}$, $\beta={fit_1['beta']:.3f}$" + "\n"
            rf"η refit:  $\log K_\mathrm{{eff}}={p['log_K_eff']:.2f}$, "
            rf"$\rho={p['rho']:.2f}$    RMSE$_{{\log L}}={fit30['rmse_logL']:.3f}$",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"empirical_eta_{SIZE_TAG}_vs_DpoverD.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"   → {out}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N, datasets = load(SIZE_TAG)
    print(f"=== {SIZE_TAG.upper()}  (N = {N:.2g}) ===")

    # Step 1 — 1-epoch fit
    s1, D1, L1 = extract_1epoch(datasets, N, scale_min=0.0)
    mask = s1 >= MIN_SCALE_1EP
    E_fixed = MULTI_N_E0 + MULTI_N_A / N ** MULTI_N_ALPHA
    print(f"E_eff(N) from multi-N joint formula = {E_fixed:.4f}")
    fit_1 = fit_1ep(D1[mask], L1[mask], E_fixed=E_fixed)
    print(f"1-epoch fit (n={int(mask.sum())}/{len(mask)}, min {MIN_SCALE_1EP}×, δ={DELTA}):")
    print(f"  E={fit_1['E']:.4f}  B={fit_1['B']:.2f}  β={fit_1['beta']:.4f}  "
          f"RMSE(logL)={fit_1['rmse_logL']:.4f}  R²={fit_1['r2_logL']:.4f}")

    # Step 2 — empirical η on multi-epoch data
    sm, D, ep, Dp, L, L_1ep = extract_multi_epoch(
        datasets, N, scale_min=0.0,
        exclude_overfit=OVERFIT_EXCLUDE.get(SIZE_TAG, set()))
    D = np.asarray(D); Dp = np.asarray(Dp); L = np.asarray(L)
    eta = per_point_eta(D, Dp, L, fit_1["E"], fit_1["B"], fit_1["beta"])
    DoN = D / N
    DpD = Dp / D
    print(f"multi-epoch points: {len(D)}, "
          f"unphysical (L < E): {int(np.isnan(eta).sum())}")

    # Step 3 — refit (log K_eff, ρ) using same fit_lse pipeline
    fit30 = fit_eta_single(D, Dp, L, N, fit_1["E"], fit_1["B"], fit_1["beta"])
    p = fit30["params"]
    print(f"η refit: log K_eff={p['log_K_eff']:.4f}  ρ={p['rho']:.4f}  "
          f"RMSE(logL)={fit30['rmse_logL']:.4f}  R²={fit30['r2_logL']:.4f}")

    # Step 4 — plots
    plot_vs_DoverN(N, DoN, DpD, eta, fit30, fit_1)
    plot_vs_DpoverD(N, DoN, DpD, eta, fit30, fit_1)
