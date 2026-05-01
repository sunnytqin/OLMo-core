"""
Same saturation diagnostics as plot_saturation_by_N.py, but for Form C
(sat × (D/N), b(N)).

Form C:   η_C = c · (D/N)^(−γ) / (1 + b · D'/D)        (per-size, 3 params)
or equivalently with N-dependent b:
          η_C = c · (D/N)^(−γ) / (1 + b₀ · (N/N_ref)^κ · D'/D)   (joint, 4 params)

Asymptote of the extra-fresh-equivalent gain:
          lim_{x→∞} η_C · x  =  c · (D/N)^(−γ) / b      (the saturation ceiling)

Same downward-trend-with-N story as Form B, but with Form C's
parameterization.  Note Form C does *not* enforce η ≤ 1, so at
small scales / large N the curve can start above 1 (this is the source
of Form C's better fit at 190M / 370M, see §3.4).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import SIZES, TTP_RATIO  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Per-size Form C fits (sat × (D/N), 3 params) — from §3.2
PER_SIZE_FORM_C = {
    "14m":  dict(c=0.70, gamma=-0.11, b=0.002),
    "30m":  dict(c=5.04, gamma=0.408, b=0.076),
    "60m":  dict(c=3.74, gamma=0.389, b=0.078),
    "190m": dict(c=4.45, gamma=0.366, b=0.156),
    "370m": dict(c=5.05, gamma=0.372, b=0.332),
}

# Joint Form C (sat × (D/N), b(N))
JOINT_FORM_C = dict(c=3.51, gamma=0.317, b0=0.062, kappa=0.542,
                    N_ref=30e6)

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 14, 12, 11, 16


def eta_form_C_per_size(x, D_over_N, c, gamma, b):
    """η = c · (D/N)^(−γ) / (1 + b · x), x = D'/D."""
    head = c * D_over_N ** (-gamma)
    return head / (1.0 + b * x)


def eta_form_C_joint(x, D_over_N, N, c, gamma, b0, kappa, N_ref):
    """η = c · (D/N)^(−γ) / (1 + b₀·(N/N_ref)^κ · x)."""
    b_eff = b0 * (N / N_ref) ** kappa
    head = c * D_over_N ** (-gamma)
    return head / (1.0 + b_eff * x)


def ceiling_per_size(D_over_N, c, gamma, b):
    """Asymptote of η·x as x → ∞."""
    return c * D_over_N ** (-gamma) / b


def ceiling_joint(D_over_N, N, c, gamma, b0, kappa, N_ref):
    b_eff = b0 * (N / N_ref) ** kappa
    return c * D_over_N ** (-gamma) / b_eff


def plot(path, scales=(0.5, 1.0, 2.0)):
    sizes = list(PER_SIZE_FORM_C.keys())
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)
    color_of = lambda i: cmap(cnorm(i))

    epochs = np.geomspace(1.5, 256, 200)
    x_grid = epochs - 1.0

    fig, axes = plt.subplots(len(scales), 3, figsize=(20, 5 * len(scales)),
                             squeeze=False)
    fig.suptitle("Form C: η_C, η_C·(D'/D), effective epochs vs epochs\n"
                 "(per-size sat × (D/N) + joint sat × (D/N), b(N))",
                 fontsize=FONT_TITLE + 1, y=1.00)

    for row, scale in enumerate(scales):
        ax_eta, ax_xtra, ax_eff = axes[row]
        ax_eta.set_title(f"η_C  (scale = {scale}×)", fontsize=FONT_TITLE)
        ax_xtra.set_title(f"η_C·(D'/D)  (scale = {scale}×)",
                          fontsize=FONT_TITLE)
        ax_eff.set_title(f"Effective epochs = 1 + η_C·(D'/D)  (scale = {scale}×)",
                         fontsize=FONT_TITLE)

        D_over_N = scale * TTP_RATIO

        for i, size in enumerate(sizes):
            N = SIZES[size][0]
            par = PER_SIZE_FORM_C[size]
            eta = eta_form_C_per_size(x_grid, D_over_N, **par)
            eta_xD = eta * x_grid
            ceil_ = ceiling_per_size(D_over_N, **par)

            color = color_of(i)
            ax_eta.plot(epochs, eta, "-", color=color, linewidth=2.2,
                        label=f"{size}  ceiling={ceil_:.1f}")
            ax_xtra.plot(epochs, eta_xD, "-", color=color, linewidth=2.2,
                         label=f"{size}  ceiling={ceil_:.1f}")
            ax_xtra.axhline(ceil_, color=color, linestyle=":", linewidth=1,
                            alpha=0.5)
            ax_eff.plot(epochs, 1.0 + eta_xD, "-", color=color, linewidth=2.2,
                        label=f"{size}")

        # Joint Form C overlay (b(N) varies — show one curve per size,
        # dashed black for clarity)
        for i, size in enumerate(sizes):
            N = SIZES[size][0]
            eta_j = eta_form_C_joint(x_grid, D_over_N, N, **JOINT_FORM_C)
            ceil_j = ceiling_joint(D_over_N, N, **JOINT_FORM_C)
            ax_eta.plot(epochs, eta_j, "--", color=color_of(i), linewidth=1.0,
                        alpha=0.6)
            ax_xtra.plot(epochs, eta_j * x_grid, "--", color=color_of(i),
                         linewidth=1.0, alpha=0.6)
            ax_eff.plot(epochs, 1.0 + eta_j * x_grid, "--", color=color_of(i),
                        linewidth=1.0, alpha=0.6)
            if i == len(sizes) - 1:  # one legend entry only
                ax_eta.plot([], [], "--", color="black", linewidth=1.4,
                            label="joint b(N) (dashed)")

        for ax in (ax_eta, ax_xtra, ax_eff):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("epochs", fontsize=FONT_LABEL)
            ax.tick_params(labelsize=FONT_TICK)
            ax.grid(alpha=0.3, which="both")
        ax_eta.set_yscale("log")
        ax_eta.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax_eta.set_ylabel("η_C", fontsize=FONT_LABEL)
        ax_xtra.set_ylabel("η_C · (D'/D)", fontsize=FONT_LABEL)
        ax_eff.set_ylabel("effective epochs", fontsize=FONT_LABEL)
        ax_eff.plot(epochs, epochs, ":", color="gray", linewidth=1, alpha=0.5)
        ax_eta.legend(fontsize=FONT_LEGEND - 1, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_ceiling_vs_N(path, scales=(0.5, 1.0, 2.0, 4.0)):
    """Form C ceiling vs N at several scales; per-size and joint b(N)."""
    sizes = list(PER_SIZE_FORM_C.keys())
    Ns = np.array([SIZES[s][0] for s in sizes])
    cmap = plt.cm.plasma
    cnorm = plt.Normalize(vmin=0, vmax=len(scales) - 1)

    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    for j, scale in enumerate(scales):
        D_over_N = scale * TTP_RATIO
        per_size = np.array([
            ceiling_per_size(D_over_N, **PER_SIZE_FORM_C[s])
            for s in sizes
        ])
        joint = np.array([
            ceiling_joint(D_over_N, SIZES[s][0], **JOINT_FORM_C)
            for s in sizes
        ])
        c = cmap(cnorm(j))
        ax.plot(Ns, per_size, "o-", color=c, linewidth=2, markersize=11,
                label=f"per-size, scale={scale}×")
        ax.plot(Ns, joint, "x--", color=c, linewidth=1.4, markersize=10,
                alpha=0.8, label=f"joint b(N), scale={scale}×")

    for s, N in zip(sizes, Ns):
        ax.annotate(s, (N, ceiling_per_size(1.0 * TTP_RATIO,
                                            **PER_SIZE_FORM_C[s])),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=FONT_LEGEND - 1, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters N", fontsize=FONT_LABEL)
    ax.set_ylabel("Form C ceiling: $c \\cdot (D/N)^{-\\gamma} / b$",
                  fontsize=FONT_LABEL)
    ax.set_title("Form C saturation ceiling vs N at several Chinchilla scales\n"
                 "(downward trend in N at fixed scale = larger models plateau sooner)",
                 fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=FONT_LEGEND - 1, loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot(os.path.join(SCRIPT_DIR, "plot_saturation_by_N_formC.pdf"))
    plot_ceiling_vs_N(os.path.join(SCRIPT_DIR, "plot_ceiling_vs_N_formC.pdf"))
