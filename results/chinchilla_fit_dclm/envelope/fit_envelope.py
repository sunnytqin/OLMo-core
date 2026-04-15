"""
Envelope scaling law for multi-epoch training (30M model).

Overview
========
Instead of modeling *how* extra epochs contribute (the η approach in ad_hoc/),
this form directly characterizes the *best achievable loss* for each fresh
data budget D when multi-epoching is allowed.

Two power laws:
    1-epoch:       L     = E + B       / D^β
    Multi-epoch:   L_min = E + B_multi / D^β_multi

where L_min(D) = min over all epoch counts of the validation loss for that D.
E (irreducible loss) is shared across both fits.

The shift between the two curves quantifies the value of multi-epoching.
We also compute an "effective data multiplier": D_eff/D, where D_eff is the
amount of fresh data that would give the same loss as multi-epoch-optimal
training with D fresh tokens.

Fitting procedure
=================
Step 1  Fit L = E + B / D^β  on 1-epoch data (scales ≥ 0.5x)
Step 2  For each D, extract L_min = min(loss over epochs)
Step 3  Fit L_min = E + B_multi / D^β_multi  with E fixed from Step 1
Step 4  Compute D_eff = (B / (L_min - E))^(1/β)  →  multiplier = D_eff / D
Step 5  Visualize
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Path to results/ for dclm_30m
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dclm_30m import ALL_DATASETS, parap_best_data

# Path to ad_hoc/ for shared fitting functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ad_hoc"))
from fit_30m import (
    N, TTP_RATIO,
    FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE,
    fmt_tokens,
    scaling_law, fit_classic,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_envelope_points(datasets):
    """For each unique D, find the minimum loss across all epoch counts.

    Returns:
        dict with keys:
            scale:      array of chinchilla scales
            D:          array of fresh token counts (one per scale)
            loss_min:   array of minimum validation loss per D
            best_epoch: array of epoch count that achieved the minimum
            all_points: list of dicts per scale, each with
                        {'D', 'scale', 'epochs', 'losses'}
    """
    scales, Ds, loss_mins, best_epochs = [], [], [], []
    all_points = []

    for ds in datasets:
        scale = ds["chinchilla_scale"][0]
        D = scale * TTP_RATIO * N
        epochs = np.array(ds["epochs"])
        losses = np.array(ds["validation_loss"], dtype=float)

        valid = ~np.isnan(losses)
        epochs_v = epochs[valid]
        losses_v = losses[valid]

        all_points.append({
            "D": D, "scale": scale,
            "epochs": epochs_v, "losses": losses_v,
        })

        idx_min = np.argmin(losses_v)
        scales.append(scale)
        Ds.append(D)
        loss_mins.append(losses_v[idx_min])
        best_epochs.append(epochs_v[idx_min])

    return {
        "scale": np.array(scales),
        "D": np.array(Ds),
        "loss_min": np.array(loss_mins),
        "best_epoch": np.array(best_epochs),
        "all_points": all_points,
    }


def compute_effective_multiplier(E, B, beta, D, L_min):
    """Compute D_eff / D for each data point.

    D_eff satisfies  E + B / D_eff^β = L_min
        → D_eff = (B / (L_min - E))^(1/β)
        → multiplier = D_eff / D
    """
    denom = L_min - E
    valid = denom > 0
    D_eff = np.full_like(D, np.nan)
    D_eff[valid] = (B / denom[valid]) ** (1.0 / beta)
    multiplier = D_eff / D
    return D_eff, multiplier


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def plot_envelope_figure(ax_fit, ax_mult,
                         D_1ep, loss_1ep, scale_1ep,
                         envelope,
                         E, B, beta, B_multi, beta_multi,
                         D_eff, multiplier,
                         fit_min_scale=0.5):
    """Populate 2 axes: power law fits (left) and effective multiplier (right).

    Args:
        ax_fit, ax_mult: matplotlib Axes
        D_1ep, loss_1ep, scale_1ep: 1-epoch data (all scales)
        envelope: dict from extract_envelope_points
        E, B, beta: 1-epoch fit parameters
        B_multi, beta_multi: envelope fit parameters
        D_eff, multiplier: from compute_effective_multiplier
        fit_min_scale: scales below this were excluded from 1-epoch fit
    """
    excluded = scale_1ep < fit_min_scale

    # ── Left panel: L vs D ──
    # Background: all multi-epoch points (faint)
    for pt in envelope["all_points"]:
        multi_mask = pt["epochs"] > 1
        if np.any(multi_mask):
            ax_fit.scatter(
                np.full(np.sum(multi_mask), pt["D"]),
                pt["losses"][multi_mask],
                s=20, color="lightgray", edgecolors="gray",
                linewidths=0.3, zorder=3, alpha=0.7,
            )

    # 1-epoch data
    ax_fit.scatter(D_1ep[excluded], loss_1ep[excluded], s=80, color="lightgray",
                   edgecolors="k", linewidths=0.5, zorder=5, label="1-ep (excluded)")
    ax_fit.scatter(D_1ep[~excluded], loss_1ep[~excluded], s=80, color="tab:blue",
                   edgecolors="k", linewidths=0.5, zorder=5, label="1-epoch data")

    # Envelope points
    ax_fit.scatter(envelope["D"], envelope["loss_min"], s=150, color="tab:red",
                   edgecolors="k", linewidths=0.8, zorder=6, marker="*",
                   label="Envelope (best epoch)")

    # Annotate envelope points with best epoch
    for i in range(len(envelope["D"])):
        ax_fit.annotate(
            f'{int(envelope["best_epoch"][i])}ep',
            (envelope["D"][i], envelope["loss_min"][i]),
            textcoords="offset points", xytext=(8, -8),
            fontsize=FONT_LEGEND - 2, color="tab:red", fontweight="bold",
        )

    # Fit curves
    D_smooth = np.geomspace(
        min(D_1ep.min(), envelope["D"].min()) * 0.5,
        max(D_1ep.max(), envelope["D"].max()) * 2,
        200,
    )
    ax_fit.plot(D_smooth, scaling_law(D_smooth, E, B, beta), "--",
                color="tab:blue", linewidth=2,
                label=f"L = {E:.2f} + {B:.0f} / D^{beta:.3f}")
    ax_fit.plot(D_smooth, scaling_law(D_smooth, E, B_multi, beta_multi), "-",
                color="tab:red", linewidth=2,
                label=f"L = {E:.2f} + {B_multi:.0f} / D^{beta_multi:.3f}")

    ax_fit.axhline(E, color="gray", linestyle=":", alpha=0.5)

    ax_fit.set_xscale("log", base=2)
    ax_fit.set_ylim(E - 0.2, np.max(loss_1ep) + 0.5)
    ax_fit.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_fit.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax_fit.set_title("1-Epoch vs Envelope Power Law", fontsize=FONT_TITLE)
    ax_fit.tick_params(labelsize=FONT_TICK)
    ax_fit.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_fit.legend(fontsize=FONT_LEGEND - 1, loc="upper right")
    ax_fit.grid(alpha=0.3)

    # ── Right panel: D_eff vs D ──
    valid = ~np.isnan(D_eff)
    ax_mult.scatter(envelope["D"][valid], D_eff[valid], s=100,
                    color="tab:red", edgecolors="k", linewidths=0.5, zorder=5)

    for i in np.where(valid)[0]:
        ax_mult.annotate(
            f'{int(envelope["best_epoch"][i])}ep ({multiplier[i]:.1f}x)',
            (envelope["D"][i], D_eff[i]),
            textcoords="offset points", xytext=(10, -5),
            fontsize=FONT_LEGEND - 1, color="tab:red", fontweight="bold",
            ha="left", va="top",
        )

    # x=y reference line
    all_D = envelope["D"][valid]
    xy_range = np.geomspace(all_D.min() * 0.5, max(all_D.max(), D_eff[valid].max()) * 1.5, 100)
    ax_mult.plot(xy_range, xy_range, "--", color="gray", linewidth=1.5, alpha=0.5,
                 label="D_eff = D (no benefit)")

    ax_mult.set_xscale("log", base=2)
    ax_mult.set_yscale("log", base=2)
    ax_mult.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_mult.set_ylabel("Effective fresh tokens D_eff", fontsize=FONT_LABEL)
    ax_mult.set_title("Effective Data from Multi-Epoching", fontsize=FONT_TITLE)
    ax_mult.tick_params(labelsize=FONT_TICK)
    ax_mult.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_mult.yaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_mult.legend(fontsize=FONT_LEGEND)
    ax_mult.grid(alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Step 1: Extract 1-epoch data and fit classic power law ──
    records_1ep = []
    for ds in ALL_DATASETS:
        scale = ds["chinchilla_scale"][0]
        D = scale * TTP_RATIO * N
        idx = ds["epochs"].index(1)
        loss = ds["validation_loss"][idx]
        if not np.isnan(loss):
            records_1ep.append((scale, D, loss))
    records_1ep = np.array(records_1ep)
    scale_1ep = records_1ep[:, 0]
    D_1ep = records_1ep[:, 1]
    loss_1ep = records_1ep[:, 2]

    fit_mask = scale_1ep >= 0.5
    E, B, beta = fit_classic(D_1ep[fit_mask], loss_1ep[fit_mask])

    # ── Step 2: Extract envelope points ──
    envelope = extract_envelope_points(ALL_DATASETS)

    # Fit envelope power law (shared E, all scales)
    _, B_multi, beta_multi = fit_classic(
        envelope["D"], envelope["loss_min"],
        E_prime=E,
    )

    # ── Step 3: Effective data multiplier (all scales) ──
    D_eff, multiplier = compute_effective_multiplier(
        E, B, beta, envelope["D"], envelope["loss_min"],
    )

    # ── Print summary table ──
    print(f"\n{'Scale':>6} {'D':>10} {'Best ep':>8} {'L_min':>8} {'D_eff':>12} {'D_eff/D':>8}")
    print("-" * 60)
    for i in range(len(envelope["D"])):
        print(f"{envelope['scale'][i]:>6.2f} {fmt_tokens(envelope['D'][i]):>10} "
              f"{int(envelope['best_epoch'][i]):>8} {envelope['loss_min'][i]:>8.3f} "
              f"{fmt_tokens(D_eff[i]) if not np.isnan(D_eff[i]) else 'N/A':>12} "
              f"{multiplier[i]:>8.1f}")

    # ── Step 4: Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plot_envelope_figure(
        ax1, ax2,
        D_1ep, loss_1ep, scale_1ep,
        envelope,
        E, B, beta, B_multi, beta_multi,
        D_eff, multiplier,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "envelope_repeat.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(SCRIPT_DIR, "envelope_repeat.png"), bbox_inches="tight", dpi=150)
    print(f"\nSaved envelope_repeat.pdf")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════
    # PARAPHRASE: solve for η using fixed E, B_multi, β_multi
    # ═══════════════════════════════════════════════════════════════════
    # D' = actual paraphrased token counts (from file sizes on disk)
    PARA_TOKEN_COUNTS = {
        0.05: 14611964,
        0.1:  29004020,
        0.5:  142688600,
        1.0:  280968222,
    }

    para_scales = np.array(parap_best_data["chinchilla_scale"], dtype=float)
    para_L_min = np.array(parap_best_data["validation_loss"], dtype=float)
    para_D = para_scales * TTP_RATIO * N
    para_Dp = np.array([PARA_TOKEN_COUNTS[s] for s in para_scales])

    # Per-point η: solve L_min = E + B_multi / (D + η·D')^β_multi
    #   → η = ((B_multi / (L_min - E))^(1/β_multi) - D) / D'
    denom = para_L_min - E
    para_eta = ((B_multi / denom) ** (1.0 / beta_multi) - para_D) / para_Dp

    # D_eff for paraphrase (invert 1-epoch law: E + B/D_eff^β = L_min)
    para_D_eff, para_mult = compute_effective_multiplier(
        E, B, beta, para_D, para_L_min,
    )

    print(f"\n{'='*60}")
    print("Paraphrase η analysis (E, B_multi, β_multi fixed from repeat)")
    print(f"E={E:.4f}, B_multi={B_multi:.2f}, β_multi={beta_multi:.4f}")
    print(f"\n{'Scale':>6} {'D':>10} {'D_para':>10} {'L_min':>8} {'η':>8} {'D_eff':>12} {'D_eff/D':>8}")
    print("-" * 70)
    for i in range(len(para_scales)):
        print(f"{para_scales[i]:>6.2f} {fmt_tokens(para_D[i]):>10} "
              f"{fmt_tokens(para_Dp[i]):>10} {para_L_min[i]:>8.3f} "
              f"{para_eta[i]:>8.3f} "
              f"{fmt_tokens(para_D_eff[i]):>12} {para_mult[i]:>8.1f}")

    # ── Figure 2: Paraphrase comparison ──
    fig2, (ax_loss, ax_deff) = plt.subplots(1, 2, figsize=(20, 8))

    # ── Left panel: L vs D (three regimes) ──
    # 1-epoch data + fit
    ax_loss.scatter(D_1ep[~fit_mask], loss_1ep[~fit_mask], s=60, color="lightgray",
                    edgecolors="k", linewidths=0.3, zorder=4)
    ax_loss.scatter(D_1ep[fit_mask], loss_1ep[fit_mask], s=80, color="tab:blue",
                    edgecolors="k", linewidths=0.5, zorder=5, label="1-epoch")
    D_smooth = np.geomspace(
        min(D_1ep.min(), envelope["D"].min()) * 0.5,
        max(D_1ep.max(), envelope["D"].max()) * 2, 200)
    ax_loss.plot(D_smooth, scaling_law(D_smooth, E, B, beta), "--",
                 color="tab:blue", linewidth=2,
                 label=f"L = {E:.2f} + {B:.0f}/D^{beta:.3f}")

    # Multi-epoch envelope + fit
    ax_loss.scatter(envelope["D"], envelope["loss_min"], s=100, color="tab:red",
                    edgecolors="k", linewidths=0.5, zorder=5, marker="s",
                    label="Repeat envelope")
    ax_loss.plot(D_smooth, scaling_law(D_smooth, E, B_multi, beta_multi), "-",
                 color="tab:red", linewidth=2,
                 label=f"L = {E:.2f} + {B_multi:.0f}/D^{beta_multi:.3f}")

    # Paraphrase envelope + curve using mean η
    ax_loss.scatter(para_D, para_L_min, s=150, color="tab:green", edgecolors="k",
                    linewidths=0.8, zorder=6, marker="*",
                    label="Paraphrase envelope")
    eta_mean = np.mean(para_eta)
    # D' ≈ 0.48 * D (paraphrased tokens scale with fresh tokens)
    D_eff_para_smooth = D_smooth + eta_mean * (0.48 * D_smooth)
    L_para_smooth = E + B_multi / D_eff_para_smooth ** beta_multi
    ax_loss.plot(D_smooth, L_para_smooth, "-", color="tab:green", linewidth=2,
                 label=f"Paraphrase (mean η={eta_mean:.2f})")

    ax_loss.axhline(E, color="gray", linestyle=":", alpha=0.3)
    ax_loss.set_xscale("log", base=2)
    ax_loss.set_ylim(E - 0.2, np.max(loss_1ep) + 0.5)
    ax_loss.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_loss.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax_loss.set_title("Scaling Laws: 1-Epoch, Repeat, Paraphrase", fontsize=FONT_TITLE)
    ax_loss.tick_params(labelsize=FONT_TICK)
    ax_loss.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_loss.legend(fontsize=FONT_LEGEND - 1, loc="upper right")
    ax_loss.grid(alpha=0.3)

    # ── Right panel: D_eff vs D (repeat + paraphrase) ──
    valid_rep = ~np.isnan(D_eff)
    ax_deff.scatter(envelope["D"][valid_rep], D_eff[valid_rep], s=100,
                    color="tab:red", edgecolors="k", linewidths=0.5, zorder=5,
                    marker="s", label="Repeat envelope")
    for i in np.where(valid_rep)[0]:
        ax_deff.annotate(
            f'{int(envelope["best_epoch"][i])}ep',
            (envelope["D"][i], D_eff[i]),
            textcoords="offset points", xytext=(8, -5),
            fontsize=FONT_LEGEND - 2, color="tab:red", ha="left")

    valid_para = ~np.isnan(para_D_eff)
    ax_deff.scatter(para_D[valid_para], para_D_eff[valid_para], s=150,
                    color="tab:green", edgecolors="k", linewidths=0.8, zorder=6,
                    marker="*", label="Paraphrase envelope")
    for i in np.where(valid_para)[0]:
        ax_deff.annotate(
            f'η={para_eta[i]:.2f}',
            (para_D[i], para_D_eff[i]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=FONT_LEGEND - 2, color="tab:green", ha="left")

    # x=y line
    all_vals = np.concatenate([envelope["D"], D_eff[valid_rep], para_D_eff[valid_para]])
    xy_range = np.geomspace(min(all_vals) * 0.3, max(all_vals) * 2, 100)
    ax_deff.plot(xy_range, xy_range, "--", color="gray", linewidth=1.5, alpha=0.5,
                 label="D_eff = D (no benefit)")

    ax_deff.set_xscale("log", base=2)
    ax_deff.set_yscale("log", base=2)
    ax_deff.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_deff.set_ylabel("Effective fresh tokens D_eff", fontsize=FONT_LABEL)
    ax_deff.set_title("Effective Data: Repeat vs Paraphrase", fontsize=FONT_TITLE)
    ax_deff.tick_params(labelsize=FONT_TICK)
    ax_deff.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_deff.yaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_deff.legend(fontsize=FONT_LEGEND - 1)
    ax_deff.grid(alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(os.path.join(SCRIPT_DIR, "envelope_paraphrase.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(SCRIPT_DIR, "envelope_paraphrase.png"), bbox_inches="tight", dpi=150)
    print(f"\nSaved envelope_paraphrase.pdf")
    plt.close(fig2)
