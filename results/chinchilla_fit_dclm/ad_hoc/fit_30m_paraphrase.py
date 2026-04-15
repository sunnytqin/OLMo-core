"""
Chinchilla scaling law fitting for paraphrase augmentation (30M model).

Same framework as fit_30m.py but for a different data augmentation strategy:
instead of repeating existing tokens, we augment with paraphrased tokens.

The scaling law is the same:

    L = E' + B / (D + η · D')^α

where:
    D   = fresh (unique) tokens available = chinchilla_scale × 20 × N
    D'  = additional tokens from the strategy = (epochs - 1) × D
    η   = discount factor for this strategy's extra tokens

The key claim: for any synthetic data strategy, we can fit η to reason about
effective tokens. Different strategies yield different η curves:
    - "repeat":    D' = repeated tokens          → η_repeat
    - "paraphrase": D' = paraphrase-augmented tokens → η_paraphrase

Fitting procedure:
    Step 1: Reuse classic fit (E', B, α) from 1-epoch repeat data
    Step 2a: Back out per-point η from paraphrase multi-epoch data
    Step 2b: Fit parametric η(D, D') on paraphrase data
    Step 3: Visualize

Note: We discard epoch=1 paraphrase points because at epoch=1 the model
has NOT seen all D fresh tokens — it saw a mix of fresh + paraphrased,
so D is not the right anchor for those points.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dclm_30m import ALL_DATASETS, parap_datasets

# ── Import shared fitting machinery from fit_30m ──
from fit_30m import (
    N, TTP_RATIO,
    FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE,
    fmt_tokens,
    scaling_law, fit_classic, compute_eta,
    eta_parametric, fit_parametric_eta,
    plot_chinchilla_fits, plot_eta_panels, plot_effective_tokens,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_repeat_1epoch():
    """Extract 1-epoch data from repeat experiments (for classic fit)."""
    records = []
    for ds in ALL_DATASETS:
        scale = ds["chinchilla_scale"][0]
        D = scale * TTP_RATIO * N
        idx = ds["epochs"].index(1)
        loss = ds["validation_loss"][idx]
        if not np.isnan(loss):
            records.append((scale, D, loss))
    records = np.array(records)
    return {
        "scale": records[:, 0],
        "D": records[:, 1],
        "loss": records[:, 2],
    }


def extract_paraphrase_data():
    """Extract paraphrase data, excluding epoch=1 points.

    At epoch=1, the model hasn't seen all D fresh tokens (it saw a mix
    of fresh + paraphrased), so D is not the correct anchor.

    Returns dict with keys: scale, D, epochs, D_prime, loss.
    D' = (epochs - 1) × D (additional tokens from the paraphrase strategy).
    """
    records = []
    for ds in parap_datasets:
        scale = ds["chinchilla_scale"][0]
        D = scale * TTP_RATIO * N
        for i, ep in enumerate(ds["epochs"]):
            if ep == 1:
                continue  # skip: model hasn't seen full D yet
            loss = ds["validation_loss"][i]
            if np.isnan(loss):
                continue
            D_prime = (ep - 1) * D
            records.append((scale, D, ep, D_prime, loss))
    records = np.array(records)
    return {
        "scale": records[:, 0],
        "D": records[:, 1],
        "epochs": records[:, 2],
        "D_prime": records[:, 3],
        "loss": records[:, 4],
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Extract data ──
    data_1ep = extract_repeat_1epoch()
    data_para = extract_paraphrase_data()

    # ── Step 1: Classic fit on 1-epoch repeat data (same as fit_30m.py) ──
    fit_mask = data_1ep["scale"] >= 0.5  # exclude data-starved scales
    E_prime, B, alpha = fit_classic(data_1ep["D"][fit_mask], data_1ep["loss"][fit_mask])

    # ── Step 2a: Per-point η for paraphrase data ──
    eta_pp = compute_eta(
        E_prime, B, alpha,
        data_para["D"], data_para["D_prime"], data_para["loss"],
    )
    print(f"\nParaphrase per-point η: median={np.nanmedian(eta_pp):.4f}, "
          f"mean={np.nanmean(eta_pp):.4f}, "
          f"min={np.nanmin(eta_pp):.4f}, max={np.nanmax(eta_pp):.4f}")

    # ── Step 2b: Parametric η for paraphrase data ──
    c, gamma, beta = fit_parametric_eta(
        data_para["D"], data_para["D_prime"], data_para["loss"],
        E_prime, B, alpha,
    )
    eta_param = eta_parametric(data_para["D"], data_para["D_prime"], c, gamma, beta)

    # ── Step 3: Plots ──
    # We need 1-epoch data arrays for the collapse plots (all scales)
    D_1ep = data_1ep["D"]
    loss_1ep = data_1ep["loss"]
    scale_1ep = data_1ep["scale"]

    # Figure 1: Chinchilla fits (3 panels)
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig1.suptitle("Paraphrase Augmentation Strategy", fontsize=FONT_TITLE + 2, y=1.02)
    plot_chinchilla_fits(
        ax1, ax2, ax3,
        D_1ep, loss_1ep, scale_1ep,
        data_para["D"], data_para["D_prime"], data_para["loss"], data_para["scale"],
        eta_pp, eta_param,
        E_prime, B, alpha, c, gamma, beta,
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(SCRIPT_DIR, "chinchilla_fits_paraphrase.pdf"), bbox_inches="tight")
    print(f"Saved chinchilla_fits_paraphrase.pdf")
    plt.close(fig1)

    # Figure 2: Eta analysis (3 panels)
    fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(27, 7))
    fig2.suptitle("Paraphrase η Analysis", fontsize=FONT_TITLE + 2, y=1.02)
    plot_eta_panels(
        ax4, ax5,
        data_para["D"], data_para["D_prime"],
        data_para["scale"], data_para["epochs"],
        eta_pp, c, gamma, beta,
    )

    plot_effective_tokens(
        ax6,
        data_para["D"], data_para["D_prime"], data_para["scale"],
        eta_pp, c, gamma, beta,
    )

    fig2.tight_layout()
    fig2.savefig(os.path.join(SCRIPT_DIR, "eta_analysis_paraphrase.pdf"), bbox_inches="tight")
    print(f"Saved eta_analysis_paraphrase.pdf")
    plt.close(fig2)

    # ── Fit repeat η for comparison ──
    from fit_30m import extract_repeat_data
    data_rep = extract_repeat_data()
    rep_multi = data_rep["epochs"] > 1
    c_rep, gamma_rep, beta_rep = fit_parametric_eta(
        data_rep["D"][rep_multi], data_rep["D_prime"][rep_multi],
        data_rep["loss"][rep_multi], E_prime, B, alpha,
    )
    # Per-point η for repeat data (for scatter points)
    eta_rep_pp = compute_eta(
        E_prime, B, alpha,
        data_rep["D"][rep_multi], data_rep["D_prime"][rep_multi],
        data_rep["loss"][rep_multi],
    )

    # ── Figure 3: Strategy comparison — max effective tokens per D ──
    fig3, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Shared scales: paraphrase only has 0.05, 0.1, 0.5, 1.0
    para_scales = sorted(set(data_para["scale"]))
    cmap_sc = plt.cm.magma_r
    norm_sc = plt.Normalize(vmin=np.log2(min(para_scales)) - 1,
                            vmax=np.log2(max(para_scales)) + 1)

    for scale in para_scales:
        D_val = scale * TTP_RATIO * N
        color = cmap_sc(norm_sc(np.log2(scale)))

        # ── Scatter: per-point η data ──
        # Paraphrase dots (circles)
        m_para = data_para["scale"] == scale
        v_para = ~np.isnan(eta_pp[m_para])
        D_p = data_para["D"][m_para][v_para]
        Dp_p = data_para["D_prime"][m_para][v_para]
        Deff_p = D_p + eta_pp[m_para][v_para] * Dp_p
        ax.scatter(D_p + Dp_p, Deff_p, s=60, color=color, marker="o",
                   edgecolors="k", linewidths=0.5, zorder=6)

        # Repeat dots (squares) — filter to matching scales
        m_rep = data_rep["scale"][rep_multi] == scale
        v_rep = ~np.isnan(eta_rep_pp[m_rep])
        D_r = data_rep["D"][rep_multi][m_rep][v_rep]
        Dp_r = data_rep["D_prime"][rep_multi][m_rep][v_rep]
        Deff_r = D_r + eta_rep_pp[m_rep][v_rep] * Dp_r
        ax.scatter(D_r + Dp_r, Deff_r, s=60, color=color, marker="s",
                   edgecolors="k", linewidths=0.5, zorder=6)

        # ── Parametric curves ──
        Dp_s = np.geomspace(D_val * 0.1, D_val * 200, 300)

        # Paraphrase (solid)
        eta_para_c = eta_parametric(D_val, Dp_s, c, gamma, beta)
        Deff_para_c = D_val + eta_para_c * Dp_s
        ax.plot(D_val + Dp_s, Deff_para_c, "-", color=color, linewidth=2.5)

        # Repeat (dashed)
        eta_rep_c = eta_parametric(D_val, Dp_s, c_rep, gamma_rep, beta_rep)
        Deff_rep_c = D_val + eta_rep_c * Dp_s
        ax.plot(D_val + Dp_s, Deff_rep_c, "--", color=color, linewidth=2.5)

        # Label the scale
        ax.text(D_val + Dp_s[-1], Deff_para_c[-1], f"  {scale}x",
                fontsize=FONT_LEGEND - 1, color=color, va="center", fontweight="bold")

    # x=y reference
    xy_min = min(para_scales) * TTP_RATIO * N
    xy_max = max(para_scales) * TTP_RATIO * N * 200
    xy_range = np.geomspace(xy_min * 0.5, xy_max, 200)
    ax.plot(xy_range, xy_range, ":", color="gray", linewidth=1.5, alpha=0.5,
            label="η=1 (all fresh)")

    # Legend entries for line/marker styles
    ax.plot([], [], "-", color="gray", linewidth=2, label="Paraphrase (solid)")
    ax.plot([], [], "--", color="gray", linewidth=2, label="Repeat (dashed)")
    ax.scatter([], [], s=60, color="gray", marker="o", edgecolors="k",
               linewidths=0.5, label="Paraphrase data")
    ax.scatter([], [], s=60, color="gray", marker="s", edgecolors="k",
               linewidths=0.5, label="Repeat data")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.set_xlabel("Total tokens trained  D + D'", fontsize=FONT_LABEL)
    ax.set_ylabel("Effective tokens  D + η·D'", fontsize=FONT_LABEL)
    ax.set_title("Strategy Comparison: Max Effective Tokens per Data Scale",
                 fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")
    ax.grid(alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(SCRIPT_DIR, "strategy_comparison.pdf"), bbox_inches="tight")
    print(f"Saved strategy_comparison.pdf")
    plt.close(fig3)
