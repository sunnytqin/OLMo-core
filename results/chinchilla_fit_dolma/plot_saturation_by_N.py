"""
Visualize "larger models saturate faster" using Form B (Muennighoff Eq 5)
parameterized as R*(D/N, N) — i.e., the saturation asymptote itself is
the explicit fitted parameter, rather than the opaque R_0.

Form B with R*(N):
    η = R*·(1 − e^{−x/R*})/x
    R* = R*_ref · (D/N / 20)^ρ · (N/N_ref)^σ
    where R*_ref is R* at the (1×, 30M) reference point.

Three panels at fixed scale (e.g., 1× Chinchilla):
  (1) η vs epochs per size — η decays faster with more epochs at large N.
  (2) η·(D'/D) vs epochs per size — extra fresh-equivalent tokens per
      fresh token; asymptote = R*, which decreases with N.
  (3) Effective epochs = 1 + η·(D'/D).

Parameters taken from the latest fit_eta.py run.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import SIZES, TTP_RATIO  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_REF = 30e6   # reference N

# Per-size Muennighoff with R*(N) form: R* = R*_ref · (D/N/20)^ρ · (N/N_ref)^σ.
# These come from per-size fits in fit_eta.py — see writeup §3.2.  R*_ref
# is the saturation budget at (1× scale, N=30M).
PER_SIZE_MUEN = {
    "14m":  dict(Rstar=22.00, rho=-0.336, sigma=-1.65),
    "30m":  dict(Rstar=21.70, rho=-1.020, sigma=-1.00),
    "60m":  dict(Rstar=12.10, rho=-0.813, sigma= 0.011),
    "190m": dict(Rstar= 5.08, rho=-1.120, sigma= 0.244),
    "370m": dict(Rstar= 5.07, rho=-0.369, sigma=-0.044),
}

# Joint Muennighoff with R*(N) — winner of joint form ranking, LOO 0.020
JOINT_MUEN = dict(Rstar=23.0, rho=-0.931, sigma=-0.69)

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 14, 12, 11, 16


def muennighoff_eta(x, R_star):
    """η = R*(1 - exp(-x/R*))/x."""
    x = np.asarray(x, dtype=np.float64)
    return R_star * (1.0 - np.exp(-x / R_star)) / x


def Rstar_of(D_over_N, N, *, Rstar, rho, sigma):
    """R*(D/N, N) = R*_ref · (D/N / 20)^ρ · (N/N_ref)^σ."""
    return Rstar * (D_over_N / 20.0) ** rho * (N / N_REF) ** sigma


def plot(path, scales=(0.5, 1.0, 2.0)):
    sizes = list(PER_SIZE_MUEN.keys())
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)
    color_of = lambda i: cmap(cnorm(i))

    epochs = np.geomspace(1.5, 256, 200)
    x_grid = epochs - 1.0   # D'/D = epochs - 1

    fig, axes = plt.subplots(len(scales), 3, figsize=(20, 5 * len(scales)),
                             squeeze=False)
    fig.suptitle("Larger models saturate faster: η, η·D'/D, and effective epochs vs epochs\n"
                 "(per-size Muennighoff fits)",
                 fontsize=FONT_TITLE + 1, y=1.00)

    for row, scale in enumerate(scales):
        ax_eta, ax_xtra, ax_eff = axes[row]
        ax_eta.set_title(f"η  (scale = {scale}×)", fontsize=FONT_TITLE)
        ax_xtra.set_title(f"η·(D'/D)  (scale = {scale}×)", fontsize=FONT_TITLE)
        ax_eff.set_title(f"Effective epochs = 1 + η·(D'/D)  (scale = {scale}×)",
                         fontsize=FONT_TITLE)

        for i, size in enumerate(sizes):
            N = SIZES[size][0]
            DoverN = scale * TTP_RATIO
            R_star = Rstar_of(DoverN, N, **PER_SIZE_MUEN[size])
            eta = muennighoff_eta(x_grid, R_star)
            eta_xD = eta * x_grid                # = R*(1 - exp(-x/R*))
            eff_epochs = 1.0 + eta_xD            # effective passes through fresh data

            color = color_of(i)
            ax_eta.plot(epochs, eta, "-", color=color, linewidth=2.2,
                        label=f"{size}  R*={R_star:.1f}")
            ax_xtra.plot(epochs, eta_xD, "-", color=color, linewidth=2.2,
                         label=f"{size}  ceiling R*={R_star:.1f}")
            ax_xtra.axhline(R_star, color=color, linestyle=":", linewidth=1,
                            alpha=0.5)
            ax_eff.plot(epochs, eff_epochs, "-", color=color, linewidth=2.2,
                        label=f"{size}")

            # Joint overlay per size (dashed, same color, thinner)
            R_star_j = Rstar_of(DoverN, N, **JOINT_MUEN)
            eta_j = muennighoff_eta(x_grid, R_star_j)
            ax_eta.plot(epochs, eta_j, "--", color=color, linewidth=1.0,
                        alpha=0.55)
            ax_xtra.plot(epochs, eta_j * x_grid, "--", color=color,
                         linewidth=1.0, alpha=0.55)
            ax_eff.plot(epochs, 1.0 + eta_j * x_grid, "--", color=color,
                        linewidth=1.0, alpha=0.55)
        # legend entry for joint (one line)
        ax_eta.plot([], [], "--", color="black", linewidth=1.2, alpha=0.7,
                    label="joint R*(N) (dashed)")

        for ax in (ax_eta, ax_xtra, ax_eff):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("epochs", fontsize=FONT_LABEL)
            ax.tick_params(labelsize=FONT_TICK)
            ax.grid(alpha=0.3, which="both")
        ax_eta.set_yscale("log")
        ax_eta.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax_eta.set_ylabel("η", fontsize=FONT_LABEL)
        ax_xtra.set_ylabel("η · (D'/D)", fontsize=FONT_LABEL)
        ax_eff.set_ylabel("effective epochs", fontsize=FONT_LABEL)
        ax_eff.plot(epochs, epochs, ":", color="gray", linewidth=1, alpha=0.5)
        ax_eta.legend(fontsize=FONT_LEGEND - 1, loc="lower left")

    # R* vs N panel — bar chart of asymptotes per size at each scale
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_Rstar_vs_N(path, scales=(0.5, 1.0, 2.0, 4.0)):
    """Standalone summary: ceiling R* (saturation parameter) vs N at several
    scales.  Decreasing trend in N at fixed scale = larger models plateau
    sooner."""
    sizes = list(PER_SIZE_MUEN.keys())
    Ns = np.array([SIZES[s][0] for s in sizes])
    cmap = plt.cm.plasma
    cnorm = plt.Normalize(vmin=0, vmax=len(scales) - 1)

    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    for j, scale in enumerate(scales):
        DoverN = scale * TTP_RATIO
        R_per_size = np.array([
            Rstar_of(DoverN, SIZES[s][0], **PER_SIZE_MUEN[s])
            for s in sizes
        ])
        R_joint = np.array([
            Rstar_of(DoverN, SIZES[s][0], **JOINT_MUEN)
            for s in sizes
        ])
        c = cmap(cnorm(j))
        ax.plot(Ns, R_per_size, "o-", color=c, linewidth=2, markersize=11,
                label=f"per-size, scale={scale}×")
        ax.plot(Ns, R_joint, "x--", color=c, linewidth=1.4, markersize=10,
                alpha=0.8, label=f"joint R*(N), scale={scale}×")

    for s, N in zip(sizes, Ns):
        ax.annotate(s, (N, Rstar_of(1.0 * TTP_RATIO, SIZES[s][0],
                                     **PER_SIZE_MUEN[s])),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=FONT_LEGEND - 1, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters N", fontsize=FONT_LABEL)
    ax.set_ylabel("Saturation parameter $R^{*} = R_{0}(D/N)^{\\rho}$",
                  fontsize=FONT_LABEL)
    ax.set_title("Saturation R* vs N at several Chinchilla scales\n"
                 "(downward trend in N = larger models plateau sooner)",
                 fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=FONT_LEGEND - 1, loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot(os.path.join(SCRIPT_DIR, "plot_saturation_by_N.pdf"))
    plot_Rstar_vs_N(os.path.join(SCRIPT_DIR, "plot_Rstar_vs_N.pdf"))
