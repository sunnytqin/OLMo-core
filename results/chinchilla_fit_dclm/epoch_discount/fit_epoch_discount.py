"""
Epoch-discount scaling law for multi-epoch training (30M model).

Overview
========
This form cleanly factors two effects:

    L = E + B / (η_epoch(e) · (D + η_para · D'))^β

where:
    η_epoch(e) = 1 + a·(e-1) / (1 + b·(e-1))
        — captures diminishing returns of repetition (any data)
        — Michaelis-Menten: η(1)=1, η(∞) = 1+a/b

    η_para = scalar discount for paraphrased tokens
        — "1 paraphrased token is worth η_para fresh tokens"

    D + η_para·D' = effective pool size in fresh-data-equivalent tokens
    η_epoch(e) scales this pool by effective passes

For repeat-only (D'=0): L = E + B / (η_epoch(e) · D)^β
For paraphrase mix:     L = E + B / (η_epoch(e) · (D + η_para·D'))^β

Fitting procedure
=================
Step 1  Fix E, B, β from classic 1-epoch fit
Step 2  Fit η_epoch (a, b) on repeat-only multi-epoch data
Step 3  Fix η_epoch, fit η_para on paraphrase data
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import least_squares

# Path to results/ for dclm_30m
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dclm_30m import ALL_DATASETS, parap_datasets

# Path to ad_hoc/ for shared fitting functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ad_hoc"))
from fit_30m import (
    N, TTP_RATIO,
    FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE,
    fmt_tokens,
    scaling_law, fit_classic,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# D' = actual paraphrased token counts (from file sizes on disk)
PARA_TOKEN_COUNTS = {
    0.05: 14611964,
    0.1:  29004020,
    0.5:  142688600,
    1.0:  280968222,
}


# ═══════════════════════════════════════════════════════════════════════
# η_epoch(e) — effective epoch multiplier (no D dependence)
# ═══════════════════════════════════════════════════════════════════════

def eta_epoch(e, D, a, gamma, b):
    """Effective epoch multiplier.

    η(e, D) = 1 + a · (D/N)^(-γ) · (e - 1) / (1 + b · (e - 1))

    At e=1: η=1.  As e→∞: η → 1 + a·(D/N)^(-γ)/b.
    Saturation ceiling is higher for small D.
    """
    em1 = e - 1.0
    return 1.0 + a * (D / N) ** (-gamma) * em1 / (1.0 + b * em1)


# ═══════════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_repeat_points():
    """Extract all (scale, D, epochs, loss) from ALL_DATASETS."""
    records = []
    for ds in ALL_DATASETS:
        scale = ds["chinchilla_scale"][0]
        D = scale * TTP_RATIO * N
        for i, ep in enumerate(ds["epochs"]):
            loss = ds["validation_loss"][i]
            if np.isnan(loss):
                continue
            records.append((scale, D, ep, loss))
    records = np.array(records)
    return {
        "scale": records[:, 0],
        "D": records[:, 1],
        "epochs": records[:, 2],
        "loss": records[:, 3],
    }


def extract_para_points():
    """Extract all (scale, D, D', epochs, loss) from parap_datasets."""
    records = []
    for ds in parap_datasets:
        scale = ds["chinchilla_scale"][0]
        D = scale * TTP_RATIO * N
        Dp = PARA_TOKEN_COUNTS[scale]
        for i, ep in enumerate(ds["epochs"]):
            loss = ds["validation_loss"][i]
            if np.isnan(loss):
                continue
            records.append((scale, D, Dp, ep, loss))
    records = np.array(records)
    return {
        "scale": records[:, 0],
        "D": records[:, 1],
        "Dp": records[:, 2],
        "epochs": records[:, 3],
        "loss": records[:, 4],
    }


# ═══════════════════════════════════════════════════════════════════════
# FITTING
# ═══════════════════════════════════════════════════════════════════════

def fit_eta_epoch(D, epochs, loss, E, B, beta):
    """Fit η_epoch(e, D) params (a, γ, b) on repeat-only data.

    L = E + B / (η_epoch(e, D) · D)^β
    """
    def predict(params):
        a, gamma, b = params
        eta = eta_epoch(epochs, D, a, gamma, b)
        return E + B / (eta * D) ** beta

    def residuals(params):
        return loss - predict(params)

    best_result = None
    best_cost = np.inf
    for a0 in [0.5, 2.0, 5.0, 10.0]:
        for g0 in [0.2, 0.5, 0.8]:
            for b0 in [0.01, 0.05, 0.2, 1.0]:
                try:
                    res = least_squares(
                        residuals, x0=[a0, g0, b0],
                        bounds=([0, 0, 0], [100, 3, 10]),
                        max_nfev=50000,
                    )
                    if res.cost < best_cost:
                        best_cost = res.cost
                        best_result = res
                except Exception:
                    pass

    a, gamma, b = best_result.x
    pred = predict(best_result.x)
    ss_res = np.sum((loss - pred) ** 2)
    ss_tot = np.sum((loss - np.mean(loss)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"η_epoch fit: a={a:.4f}, γ={gamma:.4f}, b={b:.4f}, R²={r2:.6f}")
    print(f"  η(e,D) = 1 + {a:.3f}·(D/N)^(-{gamma:.3f})·(e-1) / (1+{b:.3f}·(e-1))")
    print(f"  Saturation: η(∞,D) = 1 + {a/b:.3f}·(D/N)^(-{gamma:.3f})")
    return a, gamma, b


def fit_eta_para_fixed(pdata, E, B, beta, a, gamma, b):
    """Fit η_para (scalar) on paraphrase data, with η_epoch fixed from repeat.

    L = E + B / (η_epoch(e, D+D') · (D + η_para·D'))^β
    """
    D, Dp, ep, loss = pdata["D"], pdata["Dp"], pdata["epochs"], pdata["loss"]
    e_eff = ep * D / (D + Dp)
    valid = e_eff >= 1.0
    D_v, Dp_v, e_v, loss_v = D[valid], Dp[valid], e_eff[valid], loss[valid]
    print(f"  Fitting on {np.sum(valid)}/{len(valid)} points (passes >= 1)")

    def predict(eta_para, D_f, Dp_f, e_f):
        pool_eff = D_f + eta_para * Dp_f
        eta_e = eta_epoch(e_f, D_f + Dp_f, a, gamma, b)
        return E + B / (eta_e * pool_eff) ** beta

    def residuals(params):
        return loss_v - predict(params[0], D_v, Dp_v, e_v)

    best_result = None
    best_cost = np.inf
    for eta0 in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        try:
            res = least_squares(residuals, x0=[eta0], bounds=([0], [20]), max_nfev=10000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_result = res
        except Exception:
            pass

    eta_para = best_result.x[0]
    pred_v = predict(eta_para, D_v, Dp_v, e_v)
    ss_res = np.sum((loss_v - pred_v) ** 2)
    ss_tot = np.sum((loss_v - np.mean(loss_v)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n  [Fixed η_epoch from repeat]")
    print(f"  η_para={eta_para:.4f}, R²={r2:.6f}")
    print(f"  Mean abs error: {np.mean(np.abs(loss_v - pred_v)):.4f}")
    return eta_para


def fit_eta_para_joint(pdata, E, B, beta):
    """Jointly fit η_epoch (a_p, γ_p, b_p) + η_para on paraphrase data.

    L = E + B / (η_epoch(e, D+D'; a_p,γ_p,b_p) · (D + η_para·D'))^β
    Only E, B, β are fixed from 1-epoch.
    """
    D, Dp, ep, loss = pdata["D"], pdata["Dp"], pdata["epochs"], pdata["loss"]
    e_eff = ep * D / (D + Dp)
    valid = e_eff >= 1.0
    D_v, Dp_v, e_v, loss_v = D[valid], Dp[valid], e_eff[valid], loss[valid]
    print(f"  Fitting on {np.sum(valid)}/{len(valid)} points (passes >= 1)")

    def predict(params, D_f, Dp_f, e_f):
        a_p, gamma_p, b_p, eta_p = params
        pool_eff = D_f + eta_p * Dp_f
        eta_e = eta_epoch(e_f, D_f + Dp_f, a_p, gamma_p, b_p)
        return E + B / (eta_e * pool_eff) ** beta

    def residuals(params):
        return loss_v - predict(params, D_v, Dp_v, e_v)

    best_result = None
    best_cost = np.inf
    for a0 in [0.5, 2.0, 5.0, 10.0]:
        for g0 in [0.2, 0.5, 0.8]:
            for b0 in [0.05, 0.2, 1.0]:
                for eta0 in [0.5, 1.0, 2.0]:
                    try:
                        res = least_squares(
                            residuals, x0=[a0, g0, b0, eta0],
                            bounds=([0, 0, 0, 0], [100, 3, 10, 20]),
                            max_nfev=50000,
                        )
                        if res.cost < best_cost:
                            best_cost = res.cost
                            best_result = res
                    except Exception:
                        pass

    a_p, gamma_p, b_p, eta_para = best_result.x
    pred_v = predict(best_result.x, D_v, Dp_v, e_v)
    ss_res = np.sum((loss_v - pred_v) ** 2)
    ss_tot = np.sum((loss_v - np.mean(loss_v)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n  [Joint fit on paraphrase data]")
    print(f"  η_epoch: a={a_p:.4f}, γ={gamma_p:.4f}, b={b_p:.4f}")
    print(f"  η_epoch(e,D) = 1 + {a_p:.3f}·(D/N)^(-{gamma_p:.3f})·(e-1)/(1+{b_p:.3f}·(e-1))")
    print(f"  η_para={eta_para:.4f}")
    print(f"  R²={r2:.6f}, Mean abs error: {np.mean(np.abs(loss_v - pred_v)):.4f}")
    return a_p, gamma_p, b_p, eta_para


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def plot_repeat_panels(ax_collapse, ax_eta,
                       data, E, B, beta, a, gamma, b):
    """Two panels for repeat data: loss collapse and η vs epochs."""
    ep1 = data["epochs"] == 1
    multi = data["epochs"] > 1
    D_1ep, loss_1ep = data["D"][ep1], data["loss"][ep1]
    scale_1ep = data["scale"][ep1]
    excluded = scale_1ep < 0.5

    # ── Left: loss collapse ──
    # 1-epoch points
    ax_collapse.scatter(D_1ep[excluded], loss_1ep[excluded], s=50, color="lightgray",
                        edgecolors="k", linewidths=0.3, zorder=4)
    ax_collapse.scatter(D_1ep[~excluded], loss_1ep[~excluded], s=80, color="tab:blue",
                        edgecolors="k", linewidths=0.5, zorder=5, label="1-epoch")

    # Multi-epoch: map to D_eff = η(e)·D
    eta_vals = eta_epoch(data["epochs"][multi], data["D"][multi], a, gamma, b)
    D_eff = eta_vals * data["D"][multi]

    scales_all = sorted(set(data["scale"][multi]))
    cmap = plt.cm.magma_r
    cnorm = plt.Normalize(vmin=np.log2(min(scales_all)) - 1,
                          vmax=np.log2(max(scales_all)) + 1)
    for scale in scales_all:
        m = data["scale"][multi] == scale
        color = cmap(cnorm(np.log2(scale)))
        ax_collapse.scatter(D_eff[m], data["loss"][multi][m], s=50, color=color,
                            edgecolors="k", linewidths=0.3, zorder=5)

    all_D = np.concatenate([D_1ep, D_eff])
    D_smooth = np.geomspace(all_D.min() * 0.5, all_D.max() * 2, 200)
    ax_collapse.plot(D_smooth, scaling_law(D_smooth, E, B, beta), "-",
                     color="tab:red", linewidth=2,
                     label=f"L = {E:.2f} + {B:.0f}/D_eff^{beta:.3f}")
    ax_collapse.axhline(E, color="gray", linestyle=":", alpha=0.3)
    ax_collapse.set_xscale("log", base=2)
    ax_collapse.set_ylim(E - 0.2, np.max(loss_1ep) + 0.5)
    ax_collapse.set_xlabel("Effective tokens  η(e)·D", fontsize=FONT_LABEL)
    ax_collapse.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax_collapse.set_title("Repeat: Loss Collapse", fontsize=FONT_TITLE)
    ax_collapse.tick_params(labelsize=FONT_TICK)
    ax_collapse.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_collapse.legend(fontsize=FONT_LEGEND - 1)
    ax_collapse.grid(alpha=0.3)

    # ── Right: η vs epochs ──
    # Per-point observed η: η_obs = (B/(L-E))^(1/β) / D
    D_multi = data["D"][multi]
    loss_multi = data["loss"][multi]
    scale_multi = data["scale"][multi]
    epochs_multi = data["epochs"][multi]
    denom = loss_multi - E
    valid = denom > 0
    eta_obs = np.full_like(loss_multi, np.nan)
    eta_obs[valid] = (B / denom[valid]) ** (1.0 / beta) / D_multi[valid]

    e_smooth = np.linspace(1, 140, 200)
    # Plot one η curve per scale
    for scale in scales_all:
        D_val = scale * TTP_RATIO * N
        color = cmap(cnorm(np.log2(scale)))
        eta_curve = eta_epoch(e_smooth, D_val, a, gamma, b)
        ax_eta.plot(e_smooth, eta_curve, "-", color=color, linewidth=2, alpha=0.8)
        ax_eta.text(e_smooth[-1] + 2, eta_curve[-1], f"{scale}x",
                    fontsize=FONT_LEGEND - 2, color=color, va="center",
                    fontweight="bold")

    for scale in scales_all:
        color = cmap(cnorm(np.log2(scale)))
        m = scale_multi == scale
        v = m & valid
        ax_eta.scatter(epochs_multi[v], eta_obs[v], s=50, color=color,
                       edgecolors="k", linewidths=0.3, zorder=5)

    # Formula + R²
    r2_val = 1 - np.sum((loss_multi - (E + B / (eta_epoch(epochs_multi, D_multi, a, gamma, b) * D_multi)**beta))**2) / np.sum((loss_multi - np.mean(loss_multi))**2)
    formula = f"η(e,D) = 1+{a:.2f}·(D/N)$^{{-{gamma:.2f}}}$·(e−1)/(1+{b:.2f}·(e−1))\nR² = {r2_val:.4f}"
    ax_eta.text(0.98, 0.05, formula,
                transform=ax_eta.transAxes, fontsize=FONT_LEGEND,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax_eta.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax_eta.set_xlabel("Epochs", fontsize=FONT_LABEL)
    ax_eta.set_ylabel("η(e, D)  (effective epoch multiplier)", fontsize=FONT_LABEL)
    ax_eta.set_title("η vs Epochs by Scale", fontsize=FONT_TITLE)
    ax_eta.tick_params(labelsize=FONT_TICK)
    ax_eta.grid(alpha=0.3)


def plot_para_panels(ax_collapse, ax_deff,
                     rep_data, para_data, E, B, beta, a, gamma, b, eta_para):
    """Two panels for paraphrase: loss collapse and D_eff comparison."""
    # ── Left: paraphrase loss collapse ──
    # 1-epoch reference
    ep1 = rep_data["epochs"] == 1
    D_1ep = rep_data["D"][ep1]
    loss_1ep = rep_data["loss"][ep1]
    scale_1ep = rep_data["scale"][ep1]
    excluded = scale_1ep < 0.5
    ax_collapse.scatter(D_1ep[excluded], loss_1ep[excluded], s=40, color="lightgray",
                        edgecolors="k", linewidths=0.3, zorder=3)
    ax_collapse.scatter(D_1ep[~excluded], loss_1ep[~excluded], s=60, color="tab:blue",
                        edgecolors="k", linewidths=0.5, zorder=4, label="1-epoch")

    D, Dp, ep, loss = para_data["D"], para_data["Dp"], para_data["epochs"], para_data["loss"]
    scale = para_data["scale"]

    # D_eff = η_epoch(e_eff) · (D + η_para·D')
    e_eff = ep * D / (D + Dp)
    pool_eff = D + eta_para * Dp
    D_eff_para = eta_epoch(e_eff, D + Dp, a, gamma, b) * pool_eff

    scales_p = sorted(set(scale))
    cmap_p = plt.cm.Set2
    for idx, s in enumerate(scales_p):
        m = scale == s
        color = cmap_p(idx / max(len(scales_p) - 1, 1))
        ax_collapse.scatter(D_eff_para[m], loss[m], s=60, color=color,
                            edgecolors="k", linewidths=0.3, zorder=5,
                            label=f"Para {s}x")

    all_D = np.concatenate([D_1ep, D_eff_para])
    D_smooth = np.geomspace(all_D.min() * 0.5, all_D.max() * 2, 200)
    ax_collapse.plot(D_smooth, scaling_law(D_smooth, E, B, beta), "-",
                     color="tab:red", linewidth=2,
                     label=f"L = {E:.2f} + {B:.0f}/D_eff^{beta:.3f}")
    ax_collapse.axhline(E, color="gray", linestyle=":", alpha=0.3)

    ax_collapse.text(0.98, 0.05,
                     f"η_para = {eta_para:.3f}\n(1 para token ≈ {eta_para:.2f} fresh)",
                     transform=ax_collapse.transAxes, fontsize=FONT_LEGEND,
                     ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax_collapse.set_xscale("log", base=2)
    ax_collapse.set_ylim(E - 0.2, max(np.max(loss_1ep), np.max(loss)) + 0.5)
    ax_collapse.set_xlabel("Effective tokens  η(e)·(D+η_para·D')", fontsize=FONT_LABEL)
    ax_collapse.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax_collapse.set_title("Paraphrase: Loss Collapse", fontsize=FONT_TITLE)
    ax_collapse.tick_params(labelsize=FONT_TICK)
    ax_collapse.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_collapse.legend(fontsize=FONT_LEGEND - 2, loc="upper right")
    ax_collapse.grid(alpha=0.3)

    # ── Right: D_eff comparison (repeat vs paraphrase) ──
    # Repeat D_eff
    rep_multi = rep_data["epochs"] > 1
    rep_D = rep_data["D"][rep_multi]
    rep_ep = rep_data["epochs"][rep_multi]
    rep_scale = rep_data["scale"][rep_multi]
    rep_D_eff = eta_epoch(rep_ep, rep_D, a, gamma, b) * rep_D

    # Only show shared scales
    shared_scales = sorted(set(scale))
    for s in shared_scales:
        m = rep_scale == s
        if np.any(m):
            ax_deff.scatter(rep_D[m], rep_D_eff[m], s=40, color="tab:red",
                            edgecolors="k", linewidths=0.3, zorder=4,
                            marker="s", alpha=0.5)
    ax_deff.scatter([], [], s=40, color="tab:red", marker="s", alpha=0.5,
                    edgecolors="k", linewidths=0.3, label="Repeat")

    for idx, s in enumerate(scales_p):
        m = scale == s
        color = cmap_p(idx / max(len(scales_p) - 1, 1))
        ax_deff.scatter(D[m], D_eff_para[m], s=60, color=color,
                        edgecolors="k", linewidths=0.3, zorder=5,
                        marker="*", label=f"Para {s}x")

    all_vals = np.concatenate([rep_D, rep_D_eff, D, D_eff_para])
    xy_range = np.geomspace(min(all_vals) * 0.3, max(all_vals) * 2, 100)
    ax_deff.plot(xy_range, xy_range, "--", color="gray", linewidth=1.5, alpha=0.5,
                 label="D_eff = D")

    ax_deff.set_xscale("log", base=2)
    ax_deff.set_yscale("log", base=2)
    ax_deff.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_deff.set_ylabel("Effective tokens D_eff", fontsize=FONT_LABEL)
    ax_deff.set_title("Repeat vs Paraphrase: Effective Data", fontsize=FONT_TITLE)
    ax_deff.tick_params(labelsize=FONT_TICK)
    ax_deff.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_deff.yaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_deff.legend(fontsize=FONT_LEGEND - 2, loc="upper left")
    ax_deff.grid(alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Extract data ──
    rep_data = extract_repeat_points()
    para_data = extract_para_points()

    ep1 = rep_data["epochs"] == 1
    multi = rep_data["epochs"] > 1

    # ── Step 1: Classic 1-epoch fit ──
    fit_mask = ep1 & (rep_data["scale"] >= 0.5)
    E, B, beta = fit_classic(rep_data["D"][fit_mask], rep_data["loss"][fit_mask])

    # ── Step 2: Fit η_epoch(e) on repeat multi-epoch data ──
    a, gamma, b = fit_eta_epoch(
        rep_data["D"][multi], rep_data["epochs"][multi], rep_data["loss"][multi],
        E, B, beta,
    )

    # Print per-scale residuals
    print(f"\nPer-scale R² (repeat):")
    scales = sorted(set(rep_data["scale"][multi]))
    for scale in scales:
        m = rep_data["scale"][multi] == scale
        D_s = rep_data["D"][multi][m]
        ep_s = rep_data["epochs"][multi][m]
        loss_s = rep_data["loss"][multi][m]
        pred_s = E + B / (eta_epoch(ep_s, D_s, a, gamma, b) * D_s) ** beta
        ss_res = np.sum((loss_s - pred_s) ** 2)
        ss_tot = np.sum((loss_s - np.mean(loss_s)) ** 2)
        r2_s = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        print(f"  {scale:>5.2f}x: R² = {r2_s:.4f}")

    # ── Step 3a: Fit η_para with η_epoch fixed from repeat ──
    print(f"\n{'='*60}")
    print("Approach A: η_epoch fixed from repeat, fit η_para only")
    eta_para_fixed = fit_eta_para_fixed(para_data, E, B, beta, a, gamma, b)

    # ── Step 3b: Joint fit η_epoch + η_para on paraphrase data ──
    print(f"\n{'='*60}")
    print("Approach B: Joint fit η_epoch + η_para on paraphrase data")
    a_p, gamma_p, b_p, eta_para_joint = fit_eta_para_joint(para_data, E, B, beta)

    # ── Compare the two η_epoch functions ──
    print(f"\n{'='*60}")
    print("η_epoch comparison:")
    print(f"  Repeat:     a={a:.4f}, γ={gamma:.4f}, b={b:.4f}")
    print(f"  Paraphrase: a={a_p:.4f}, γ={gamma_p:.4f}, b={b_p:.4f}")
    print(f"\n  η_para (fixed η_epoch): {eta_para_fixed:.4f}")
    print(f"  η_para (joint fit):     {eta_para_joint:.4f}")

    # ── Step 4: Plots ──
    # Figure 1: Repeat (2 panels) — unchanged
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plot_repeat_panels(ax1, ax2, rep_data, E, B, beta, a, gamma, b)
    fig1.tight_layout()
    fig1.savefig(os.path.join(SCRIPT_DIR, "epoch_discount.pdf"), bbox_inches="tight")
    fig1.savefig(os.path.join(SCRIPT_DIR, "epoch_discount.png"), bbox_inches="tight", dpi=150)
    print(f"\nSaved epoch_discount.pdf")
    plt.close(fig1)

    # Figure 2: Paraphrase with joint fit (2 panels)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))
    plot_para_panels(ax3, ax4, rep_data, para_data, E, B, beta,
                     a_p, gamma_p, b_p, eta_para_joint)
    fig2.tight_layout()
    fig2.savefig(os.path.join(SCRIPT_DIR, "epoch_discount_paraphrase.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(SCRIPT_DIR, "epoch_discount_paraphrase.png"), bbox_inches="tight", dpi=150)
    print(f"\nSaved epoch_discount_paraphrase.pdf")
    plt.close(fig2)

    # Figure 3: η_epoch comparison (repeat vs paraphrase)
    fig3, ax_cmp = plt.subplots(1, 1, figsize=(12, 8))
    e_smooth = np.linspace(1, 100, 200)
    # Pick a few representative pool sizes
    pool_sizes = {
        "45M (0.05x para)": 45e6,
        "89M (0.1x para)": 89e6,
        "443M (0.5x para)": 443e6,
        "881M (1.0x para)": 881e6,
    }
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(pool_sizes)))
    for (label, pool_sz), color in zip(pool_sizes.items(), colors):
        eta_rep = eta_epoch(e_smooth, pool_sz, a, gamma, b)
        eta_par = eta_epoch(e_smooth, pool_sz, a_p, gamma_p, b_p)
        ax_cmp.plot(e_smooth, eta_rep, "--", color=color, linewidth=2, alpha=0.7)
        ax_cmp.plot(e_smooth, eta_par, "-", color=color, linewidth=2, alpha=0.9,
                    label=label)

    # Legend entries for line styles
    ax_cmp.plot([], [], "--", color="gray", linewidth=2, label="Repeat η_epoch")
    ax_cmp.plot([], [], "-", color="gray", linewidth=2, label="Paraphrase η_epoch")

    ax_cmp.text(0.98, 0.05,
                f"Repeat: a={a:.2f}, γ={gamma:.2f}, b={b:.2f}\n"
                f"Para:   a={a_p:.2f}, γ={gamma_p:.2f}, b={b_p:.2f}\n"
                f"η_para (joint) = {eta_para_joint:.3f}",
                transform=ax_cmp.transAxes, fontsize=FONT_LEGEND,
                ha="right", va="bottom", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax_cmp.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.4)
    ax_cmp.set_xlabel("Passes through dataset", fontsize=FONT_LABEL)
    ax_cmp.set_ylabel("η_epoch  (effective epoch multiplier)", fontsize=FONT_LABEL)
    ax_cmp.set_title("η_epoch: Repeat vs Paraphrase (dashed vs solid)", fontsize=FONT_TITLE)
    ax_cmp.tick_params(labelsize=FONT_TICK)
    ax_cmp.legend(fontsize=FONT_LEGEND - 1, loc="upper left")
    ax_cmp.grid(alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(SCRIPT_DIR, "eta_epoch_comparison.pdf"), bbox_inches="tight")
    fig3.savefig(os.path.join(SCRIPT_DIR, "eta_epoch_comparison.png"), bbox_inches="tight", dpi=150)
    print(f"Saved eta_epoch_comparison.pdf")
    plt.close(fig3)
