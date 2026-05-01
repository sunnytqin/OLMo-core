"""
Intuition plots for the three η forms (A, B, C) from writeup.md §3.3.

Each row varies one input (D'/D, D/N, N, epochs) and overlays all three
forms or sweeps a parameter, so you can read off:
  - the basic shape of η as a function of repeats / overtraining / size,
  - the tail of η · D'/D (peak-then-decay vs plateau vs saturation),
  - how N shifts the curves at a fixed scale.

Joint-fit parameters (writeup.md §3.3):
  A: η₀ = 0.946, R₀ = 471, ρ = -0.755     (3 par, η ≤ 1 always)
  B: R₀ = 207,    ρ = -0.834               (2 par, Muennighoff '23 Eq 5)
  C: c = 3.51, γ = 0.317, b₀ = 0.062,
     κ = 0.54, N_ref = 30M                 (4 par, can have η > 1)
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from data import SIZES, TTP_RATIO  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────── joint fits from writeup §3.3 ───────────
PARAMS_A = dict(eta0=0.946, R0=471.0, rho=-0.755)
PARAMS_B = dict(R0=207.0, rho=-0.834)
PARAMS_C = dict(c=3.51, gamma=0.317, b0=0.062, kappa=0.54, N_ref=30e6)


def eta_A(x, DoverN, *, eta0, R0, rho):
    """Form A: η₀ exp(−x / (R₀ (D/N)^ρ)).  x = D'/D."""
    R_star = R0 * DoverN ** rho
    return eta0 * np.exp(-x / R_star)


def eta_B(x, DoverN, *, R0, rho):
    """Form B (Muennighoff Eq 5): R*(1 − e^{−x/R*})/x.  Limit x→0 = 1."""
    R_star = R0 * np.asarray(DoverN, dtype=np.float64) ** rho
    x = np.asarray(x, dtype=np.float64)
    x_b, R_b = np.broadcast_arrays(x, R_star)
    out = np.where(
        x_b < 1e-8,
        1.0,
        R_b * (1.0 - np.exp(-x_b / np.where(R_b == 0, 1e-30, R_b))) /
        np.where(x_b == 0, 1e-30, x_b),
    )
    return out


def eta_C(x, DoverN, N, *, c, gamma, b0, kappa, N_ref):
    """Form C: c (D/N)^{-γ} / (1 + b₀ (N/N_ref)^κ x)."""
    b_eff = b0 * (N / N_ref) ** kappa
    return c * DoverN ** (-gamma) / (1.0 + b_eff * x)


def call_eta(form, x, DoverN, N):
    if form == "A":
        return eta_A(x, DoverN, **PARAMS_A)
    if form == "B":
        return eta_B(x, DoverN, **PARAMS_B)
    if form == "C":
        return eta_C(x, DoverN, N, **PARAMS_C)
    raise ValueError(form)


# ─────────── styling ───────────
SIZES_TO_PLOT = ["14m", "30m", "60m", "190m", "370m"]
N_VALUES = {s: SIZES[s][0] for s in SIZES_TO_PLOT}
SIZE_COLORS = {
    s: plt.cm.viridis(i / (len(SIZES_TO_PLOT) - 1))
    for i, s in enumerate(SIZES_TO_PLOT)
}
FORM_COLORS = {"A": "tab:blue", "B": "tab:red", "C": "tab:green"}
FORM_LABELS = {
    "A": "A: exp(D'/D), R(D/N)",
    "B": "B: Muennighoff Eq 5",
    "C": "C: sat × (D/N), b(N)",
}
FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 12, 10, 9, 13


# ─────────── Figure 1: 3 cols (forms) × 3 rows (axes) ───────────
def figure_form_grid(path, scale=1.0):
    """For each form, three diagnostic panels:
        row 1: η vs D'/D at scale=1×, one curve per N
        row 2: η·(D'/D) vs D'/D at scale=1× — fresh-equiv. extra tokens
        row 3: η vs D/N (sweeping scale) at fixed epochs=4 (D'/D=3)
    """
    forms = ["A", "B", "C"]

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(f"η across the three forms — fixed scale = {scale}× Chinchilla "
                 f"(D/N = {scale*TTP_RATIO:.0f})\n"
                 "rows: η vs D'/D, η·D'/D vs D'/D, η vs D/N (epochs = 4)",
                 fontsize=FONT_TITLE + 1, y=1.00)

    epochs = np.geomspace(1.05, 256, 300)
    x_grid = epochs - 1.0
    DoverN_fixed = scale * TTP_RATIO

    # row 1: η vs D'/D
    for j, form in enumerate(forms):
        ax = axes[0, j]
        for s in SIZES_TO_PLOT:
            N = N_VALUES[s]
            eta = call_eta(form, x_grid, DoverN_fixed, N)
            ax.plot(x_grid, eta, "-", color=SIZE_COLORS[s], lw=2.0,
                    label=f"{s}")
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("D'/D  (= epochs − 1)", fontsize=FONT_LABEL)
        ax.set_ylabel("η", fontsize=FONT_LABEL)
        ax.set_title(f"Form {form}: {FORM_LABELS[form]}",
                     fontsize=FONT_TITLE - 1)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=FONT_TICK)
        if j == 0:
            ax.legend(fontsize=FONT_LEGEND, title="N", loc="best")

    # row 2: η · D'/D  vs D'/D — the "fresh-equivalent extra tokens" curve
    for j, form in enumerate(forms):
        ax = axes[1, j]
        for s in SIZES_TO_PLOT:
            N = N_VALUES[s]
            eta = call_eta(form, x_grid, DoverN_fixed, N)
            ax.plot(x_grid, eta * x_grid, "-", color=SIZE_COLORS[s], lw=2.0,
                    label=f"{s}")
        ax.set_xscale("log")
        ax.set_xlabel("D'/D", fontsize=FONT_LABEL)
        ax.set_ylabel("η · D'/D   (extra fresh-equivalent token ratio)",
                      fontsize=FONT_LABEL)
        ax.set_title(f"Form {form}: {'peak-then-decay' if form=='A' else 'plateau' if form=='B' else 'saturation'}",
                     fontsize=FONT_TITLE - 1)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=FONT_TICK)
        if j == 0:
            ax.legend(fontsize=FONT_LEGEND, title="N", loc="best")

    # row 3: η vs D/N at fixed epochs=4
    epochs_fixed = 4
    x_fixed = epochs_fixed - 1.0   # D'/D = 3
    DoverN_grid = np.geomspace(2, 500, 300)
    for j, form in enumerate(forms):
        ax = axes[2, j]
        for s in SIZES_TO_PLOT:
            N = N_VALUES[s]
            eta = call_eta(form, x_fixed, DoverN_grid, N)
            ax.plot(DoverN_grid, eta, "-", color=SIZE_COLORS[s], lw=2.0,
                    label=f"{s}")
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("D/N  (overtraining ratio)", fontsize=FONT_LABEL)
        ax.set_ylabel(f"η at epochs={epochs_fixed}  (D'/D = {x_fixed:.0f})",
                      fontsize=FONT_LABEL)
        ax.set_title(f"Form {form}: η vs D/N", fontsize=FONT_TITLE - 1)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=FONT_TICK)
        # Mark Chinchilla scales for reference
        for s_mark, lbl in [(0.5, "0.5×"), (1.0, "1×"), (4.0, "4×"),
                             (16.0, "16×")]:
            DoN = s_mark * TTP_RATIO
            ax.axvline(DoN, color="lightgray", ls=":", lw=0.8, alpha=0.6)
            ax.text(DoN, ax.get_ylim()[1] * 0.95, lbl,
                    rotation=90, va="top", ha="right",
                    fontsize=FONT_LEGEND - 1, color="gray")
        if j == 0:
            ax.legend(fontsize=FONT_LEGEND, title="N", loc="best")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ─────────── Figure 2: head-to-head, all three forms on same axes ───────────
def figure_head_to_head(path):
    """All three forms overlaid for a few representative (N, scale) cuts.
    Rows: scales (0.5×, 1×, 4×).  Cols: η vs D'/D | η·D'/D vs D'/D | η vs N.
    Within each panel, the three forms plot in different colors; sizes are
    rendered as line styles or separate panels as needed.
    """
    scales = [0.5, 1.0, 4.0]
    forms = ["A", "B", "C"]

    epochs = np.geomspace(1.05, 256, 300)
    x_grid = epochs - 1.0

    fig, axes = plt.subplots(len(scales), 3, figsize=(17, 4.6 * len(scales)))
    fig.suptitle("Head-to-head: forms A / B / C overlaid\n"
                 "rows = Chinchilla scale, cols = η vs D'/D, η·D'/D, η vs N",
                 fontsize=FONT_TITLE + 1, y=1.00)

    for row, scale in enumerate(scales):
        DoverN = scale * TTP_RATIO

        # col 0: η vs D'/D — pick mid size 30M and 370M, two line styles
        ax = axes[row, 0]
        for s, ls in [("30m", "-"), ("370m", "--")]:
            N = N_VALUES[s]
            for form in forms:
                eta = call_eta(form, x_grid, DoverN, N)
                ax.plot(x_grid, eta, ls=ls, color=FORM_COLORS[form], lw=1.8)
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("D'/D", fontsize=FONT_LABEL)
        ax.set_ylabel("η", fontsize=FONT_LABEL)
        ax.set_title(f"η vs D'/D  (scale={scale}×)", fontsize=FONT_TITLE - 1)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=FONT_TICK)
        if row == 0:
            handles = [Line2D([0], [0], color=FORM_COLORS[f], lw=1.8,
                              label=FORM_LABELS[f]) for f in forms]
            handles += [Line2D([0], [0], color="black", lw=1.8, ls="-",
                               label="N=30M"),
                        Line2D([0], [0], color="black", lw=1.8, ls="--",
                               label="N=370M")]
            ax.legend(handles=handles, fontsize=FONT_LEGEND, loc="best")

        # col 1: η · D'/D vs D'/D — same cuts
        ax = axes[row, 1]
        for s, ls in [("30m", "-"), ("370m", "--")]:
            N = N_VALUES[s]
            for form in forms:
                eta = call_eta(form, x_grid, DoverN, N)
                ax.plot(x_grid, eta * x_grid, ls=ls,
                        color=FORM_COLORS[form], lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("D'/D", fontsize=FONT_LABEL)
        ax.set_ylabel("η · D'/D", fontsize=FONT_LABEL)
        ax.set_title(f"η·D'/D  (scale={scale}×)\n"
                     "A peaks-and-decays, B plateaus, C saturates differently",
                     fontsize=FONT_TITLE - 1)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=FONT_TICK)

        # col 2: η vs N at fixed epochs (4ep, so D'/D = 3) at this scale
        ax = axes[row, 2]
        N_grid = np.geomspace(10e6, 1e9, 200)
        x_fixed = 3.0
        for form in forms:
            etas = np.array([call_eta(form, x_fixed, DoverN, N) for N in N_grid])
            ax.plot(N_grid / 1e6, etas, "-", color=FORM_COLORS[form], lw=2.0,
                    label=FORM_LABELS[form])
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        # show the actual fitted sizes as ticks
        for s in SIZES_TO_PLOT:
            ax.axvline(N_VALUES[s] / 1e6, color="lightgray", lw=0.7, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("N  (model parameters, M)", fontsize=FONT_LABEL)
        ax.set_ylabel("η at epochs=4", fontsize=FONT_LABEL)
        ax.set_title(f"η vs N at scale={scale}×, epochs=4",
                     fontsize=FONT_TITLE - 1)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=FONT_TICK)
        if row == 0:
            ax.legend(fontsize=FONT_LEGEND, loc="best")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ─────────── Figure 3: η · D'  (raw extra fresh tokens) ───────────
def figure_extra_tokens(path, scale=1.0):
    """A possibly more practitioner-friendly view: how many *fresh-equivalent
    tokens* does η · D' add at each scale?  This is the quantity that enters
    D_eff = D + η·D'.  Plot at fixed scale, varying N.

    Three columns (forms) × two rows: η·D' vs epochs and D_eff/D vs epochs.
    """
    forms = ["A", "B", "C"]
    epochs = np.geomspace(1.05, 256, 300)
    x_grid = epochs - 1.0
    DoverN = scale * TTP_RATIO

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Effective-token bookkeeping at scale = {scale}× Chinchilla\n"
                 "row 1: extra fresh-equivalent tokens (η · D').  "
                 "row 2: total fresh-equivalent multiplier D_eff/D = 1 + η·D'/D",
                 fontsize=FONT_TITLE + 1, y=1.00)

    for j, form in enumerate(forms):
        ax_top = axes[0, j]
        ax_bot = axes[1, j]
        for s in SIZES_TO_PLOT:
            N = N_VALUES[s]
            D = scale * TTP_RATIO * N
            eta = call_eta(form, x_grid, DoverN, N)
            extra_tokens = eta * x_grid * D     # = η · D'
            mult = 1.0 + eta * x_grid           # = D_eff / D
            ax_top.plot(epochs, extra_tokens, "-",
                        color=SIZE_COLORS[s], lw=2.0, label=s)
            ax_bot.plot(epochs, mult, "-",
                        color=SIZE_COLORS[s], lw=2.0, label=s)

        ax_top.set_xscale("log", base=2)
        ax_top.set_yscale("log")
        ax_top.set_xlabel("epochs", fontsize=FONT_LABEL)
        ax_top.set_ylabel("η · D'   (fresh-equivalent extra tokens)",
                          fontsize=FONT_LABEL)
        ax_top.set_title(f"Form {form}", fontsize=FONT_TITLE - 1)
        ax_top.grid(alpha=0.3, which="both")
        ax_top.tick_params(labelsize=FONT_TICK)

        ax_bot.set_xscale("log", base=2)
        ax_bot.set_xlabel("epochs", fontsize=FONT_LABEL)
        ax_bot.set_ylabel("D_eff / D", fontsize=FONT_LABEL)
        ax_bot.plot(epochs, epochs, ":", color="gray", lw=1,
                    label="naive (1 epoch worth/repeat)")
        ax_bot.grid(alpha=0.3, which="both")
        ax_bot.tick_params(labelsize=FONT_TICK)
        if j == 0:
            ax_top.legend(fontsize=FONT_LEGEND, title="N", loc="best")
            ax_bot.legend(fontsize=FONT_LEGEND, title="N", loc="best")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    figure_form_grid(os.path.join(SCRIPT_DIR, "plot_eta_intuition_grid.pdf"))
    figure_head_to_head(os.path.join(SCRIPT_DIR, "plot_eta_intuition_h2h.pdf"))
    figure_extra_tokens(os.path.join(SCRIPT_DIR,
                                     "plot_eta_intuition_extra_tokens.pdf"))
