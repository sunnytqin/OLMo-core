"""
Standalone overlay: fig4_saturation_curves with empirical per-point η scatter.

Does NOT re-run the joint fit pipeline. Canonical k=15 anchors from
writeup_final.md §2 are hard-coded below; only the empirical (per-point ΔL
solved) η values are computed at runtime by reusing the same helpers fig2
already calls.

Empirical points are filtered to scale ≈ SCALE_REF so they line up with the
model curves. Set ONLY_SCALE_1 = False to overlay all scales instead.

Output: test/fig4_saturation_curves_with_empirical_{scale}x.pdf
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Allow imports from the parent fit dir (data.py, fit_eta.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data import SIZES, TTP_RATIO  # noqa: E402
from fit_eta import collect_multi_epoch_all_sizes, per_point_eta_diff  # noqa: E402


# ── Canonical one-shot k=15 anchors (writeup_final.md §2) ────────────
CHINCH = dict(E=0.050, A=31.5, B=16_539.0, alpha=0.137, beta=0.436)
ETA_PARAMS = dict(log_K=10.32, rho=-0.270, sigma=-0.388)

# Filter empirical points to ≈ this Chinchilla scale before overlaying
ONLY_SCALE_1 = True
SCALE_REF = 0.5  # default scale — overridden by SCALES loop in __main__
SCALE_TOL = 0.05  # relative — scale within ±5 % of SCALE_REF
SCALES = (0.05, 0.1, 0.5, 1.0, 2.0, 4.0)

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


def cmap_for(n):
    return plt.cm.viridis(np.linspace(0, 1, n))


def Rstar_KN(D, N, log_K, rho, sigma):
    return np.exp(log_K + rho * np.log(D / N) + sigma * np.log(N))


def main(scale=1.0):
    log_K, rho, sigma = ETA_PARAMS["log_K"], ETA_PARAMS["rho"], ETA_PARAMS["sigma"]
    B, beta = CHINCH["B"], CHINCH["beta"]

    # ── empirical per-point η via ΔL (same call fig2 uses) ───────────
    # scale_min=0.0 so the smallest scales (0.05, 0.1) are not excluded.
    tags, Ns, scales, Ds, eps_emp, Dps, Ls, L_1eps = collect_multi_epoch_all_sizes(0.0)
    eta_pp = per_point_eta_diff(Ds, Dps, Ls, L_1eps, B, beta)
    valid = ~np.isnan(eta_pp) & (eta_pp >= 0) & (eta_pp <= 5)
    if ONLY_SCALE_1:
        valid &= np.abs(np.asarray(scales) - scale) <= SCALE_TOL * scale
    print(f"scale={scale}×: empirical points kept: {int(valid.sum())} / {len(valid)}"
          + (f"  (within ±{SCALE_TOL:.0%} of {scale}×)" if ONLY_SCALE_1 else ""))

    sizes_with_data = ["14m", "30m", "60m", "190m", "370m"]
    sizes_plot = sorted(sizes_with_data, key=lambda t: SIZES[t][0])
    colors = {t: c for t, c in zip(sizes_plot, cmap_for(len(sizes_plot)))}

    epochs = np.geomspace(1.5, 256, 200)
    x = epochs - 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_eta, ax_xtra = axes

    # ── model curves (same as paper_figures.fig4_saturation_curves) ──
    for tag in sizes_plot:
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

    # ── empirical scatter overlay ────────────────────────────────────
    for tag in sizes_plot:
        m = (tags == tag) & valid
        if not m.any():
            continue
        x_emp = Dps[m] / Ds[m]                # = D'/D = epochs - 1
        epochs_emp = 1.0 + x_emp
        eta_emp = eta_pp[m]
        ax_eta.scatter(epochs_emp, eta_emp, s=42, color=colors[tag],
                       edgecolors='k', linewidths=0.4, alpha=0.85, zorder=3)
        ax_xtra.scatter(epochs_emp, eta_emp * x_emp, s=42, color=colors[tag],
                        edgecolors='k', linewidths=0.4, alpha=0.85, zorder=3)

    ax_eta.set_xscale('log', base=2)
    ax_eta.set_yscale('log')
    ax_eta.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax_eta.set_xlabel("epochs")
    ax_eta.set_ylabel(r"$\eta$")
    title_a = rf"(a) $\eta$ vs epochs at scale ${scale}\times$  (• empirical)"
    ax_eta.set_title(title_a)
    ax_eta.legend(loc='lower left', ncol=1)

    ax_xtra.set_xscale('log', base=2)
    ax_xtra.set_xlabel("epochs")
    ax_xtra.set_ylabel(r"$\eta \cdot D'/D$  (extra fresh-equivalent tokens / $D$)")
    ax_xtra.set_title(r"(b) saturation: dotted = $R^*$ asymptote   (• empirical)")
    ax_xtra.legend(loc='lower right')

    fig.tight_layout()
    scale_tag = f"{scale:g}".replace(".", "p")
    out = os.path.join(OUT_DIR, f"fig4_saturation_curves_with_empirical_{scale_tag}x.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    for s in SCALES:
        main(scale=s)
