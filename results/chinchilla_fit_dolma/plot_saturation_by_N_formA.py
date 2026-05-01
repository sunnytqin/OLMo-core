"""
Same saturation diagnostics as plot_saturation_by_N.py (Form B), but for
Form A (the user-proposed exponential decay form).

Form A:   η_A = η_0 · exp(−(D'/D) / R)        with R = R_0 · (D/N)^ρ.

Unlike Forms B and C, Form A does NOT plateau — η_A·(D'/D) has a
maximum at x = R and then decays back to zero:

    η_A · x  =  η_0 · x · e^{−x/R}
    peak     :  x = R,  value = η_0 · R / e
    x → ∞    :  η_A · x → 0  (extra fresh-equivalent tokens vanish)

So "epochs to plateau" doesn't quite apply; the analogous quantity is
"epochs to peak benefit" = 1 + R, which decreases with N at fixed
scale (same headline as B and C).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import SIZES, TTP_RATIO  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Per-size Form A fits — from §3.2.  η_A = η_0 · exp(−x / R),
#   R = R_0 · (D/N)^ρ.
PER_SIZE_FORM_A = {
    "14m":  dict(eta0=0.917, R0=535.0, rho=-0.791),
    "30m":  dict(eta0=0.920, R0=501.0, rho=-0.766),
    "60m":  dict(eta0=0.783, R0=281.0, rho=-0.643),
    "190m": dict(eta0=1.410, R0=322.0, rho=-1.080),
    "370m": dict(eta0=1.210, R0=128.0, rho=-0.890),
}

# Joint Form A
JOINT_FORM_A = dict(eta0=0.946, R0=471.0, rho=-0.755)

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 14, 12, 11, 16


def eta_form_A(x, D_over_N, eta0, R0, rho):
    """η = η0 · exp(−x / R(D/N)),  R = R0 · (D/N)^ρ."""
    R = R0 * D_over_N ** rho
    return eta0 * np.exp(-x / R)


def peak_form_A(D_over_N, eta0, R0, rho):
    """η · x is maximized at x = R; peak value = eta0 · R / e."""
    R = R0 * D_over_N ** rho
    return R, eta0 * R / np.e


def plot(path, scales=(0.5, 1.0, 2.0)):
    sizes = list(PER_SIZE_FORM_A.keys())
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)
    color_of = lambda i: cmap(cnorm(i))

    epochs = np.geomspace(1.5, 256, 200)
    x_grid = epochs - 1.0

    fig, axes = plt.subplots(len(scales), 3, figsize=(20, 5 * len(scales)),
                             squeeze=False)
    fig.suptitle("Form A: η_A, η_A·(D'/D), effective epochs vs epochs\n"
                 "(per-size exp form;  η_A·x has a peak at x=R, not a plateau)",
                 fontsize=FONT_TITLE + 1, y=1.00)

    for row, scale in enumerate(scales):
        ax_eta, ax_xtra, ax_eff = axes[row]
        ax_eta.set_title(f"η_A  (scale = {scale}×)", fontsize=FONT_TITLE)
        ax_xtra.set_title(f"η_A·(D'/D)  (scale = {scale}×)",
                          fontsize=FONT_TITLE)
        ax_eff.set_title(f"Effective epochs = 1 + η_A·(D'/D)  (scale = {scale}×)",
                         fontsize=FONT_TITLE)

        D_over_N = scale * TTP_RATIO

        for i, size in enumerate(sizes):
            par = PER_SIZE_FORM_A[size]
            eta = eta_form_A(x_grid, D_over_N, **par)
            eta_xD = eta * x_grid
            R_loc, peak = peak_form_A(D_over_N, **par)

            color = color_of(i)
            ax_eta.plot(epochs, eta, "-", color=color, linewidth=2.2,
                        label=f"{size}  R={R_loc:.1f}")
            ax_xtra.plot(epochs, eta_xD, "-", color=color, linewidth=2.2,
                         label=f"{size}  peak={peak:.1f} @ ep={1+R_loc:.0f}")
            # mark peak with a vertical dashed line at x = R
            ax_xtra.axvline(1 + R_loc, color=color, linestyle=":",
                            linewidth=1, alpha=0.5)
            ax_eff.plot(epochs, 1.0 + eta_xD, "-", color=color, linewidth=2.2,
                        label=f"{size}")

        # Joint Form A overlay (one line; same η_0, R_0, ρ for all sizes,
        # but R depends on D/N so it's the same at fixed scale)
        eta_j = eta_form_A(x_grid, D_over_N, **JOINT_FORM_A)
        R_j, peak_j = peak_form_A(D_over_N, **JOINT_FORM_A)
        ax_eta.plot(epochs, eta_j, "--", color="black", linewidth=1.4,
                    alpha=0.8, label=f"joint  R={R_j:.1f}")
        ax_xtra.plot(epochs, eta_j * x_grid, "--", color="black",
                     linewidth=1.4, alpha=0.8,
                     label=f"joint  peak={peak_j:.1f}")
        ax_eff.plot(epochs, 1.0 + eta_j * x_grid, "--", color="black",
                    linewidth=1.4, alpha=0.8, label="joint")

        for ax in (ax_eta, ax_xtra, ax_eff):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("epochs", fontsize=FONT_LABEL)
            ax.tick_params(labelsize=FONT_TICK)
            ax.grid(alpha=0.3, which="both")
        ax_eta.set_yscale("log")
        ax_eta.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax_eta.set_ylabel("η_A", fontsize=FONT_LABEL)
        ax_xtra.set_ylabel("η_A · (D'/D)", fontsize=FONT_LABEL)
        ax_eff.set_ylabel("effective epochs", fontsize=FONT_LABEL)
        ax_eff.plot(epochs, epochs, ":", color="gray", linewidth=1, alpha=0.5)
        ax_eta.legend(fontsize=FONT_LEGEND - 1, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_R_vs_N(path, scales=(0.5, 1.0, 2.0, 4.0)):
    """Form A 'epochs to peak benefit' (=1+R) vs N at several scales."""
    sizes = list(PER_SIZE_FORM_A.keys())
    Ns = np.array([SIZES[s][0] for s in sizes])
    cmap = plt.cm.plasma
    cnorm = plt.Normalize(vmin=0, vmax=len(scales) - 1)

    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    for j, scale in enumerate(scales):
        D_over_N = scale * TTP_RATIO
        R_per_size = np.array([
            peak_form_A(D_over_N, **PER_SIZE_FORM_A[s])[0]
            for s in sizes
        ])
        R_joint = peak_form_A(D_over_N, **JOINT_FORM_A)[0]
        c = cmap(cnorm(j))
        ax.plot(Ns, R_per_size, "o-", color=c, linewidth=2, markersize=11,
                label=f"per-size, scale={scale}×")
        ax.axhline(R_joint, color=c, linestyle="--", linewidth=1.2,
                   alpha=0.7,
                   label=f"joint, scale={scale}×: R={R_joint:.1f}")

    for s, N in zip(sizes, Ns):
        ax.annotate(s, (N, peak_form_A(1.0 * TTP_RATIO, **PER_SIZE_FORM_A[s])[0]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=FONT_LEGEND - 1, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters N", fontsize=FONT_LABEL)
    ax.set_ylabel("Form A R = $R_0 \\cdot (D/N)^{\\rho}$  (epochs-to-peak ≈ 1+R)",
                  fontsize=FONT_LABEL)
    ax.set_title("Form A scale parameter R vs N at several Chinchilla scales\n"
                 "(downward trend in N at fixed scale = larger models peak sooner)",
                 fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=FONT_LEGEND - 1, loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot(os.path.join(SCRIPT_DIR, "plot_saturation_by_N_formA.pdf"))
    plot_R_vs_N(os.path.join(SCRIPT_DIR, "plot_R_vs_N_formA.pdf"))
