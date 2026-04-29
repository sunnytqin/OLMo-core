"""
Chinchilla scaling law fitting for multi-epoch training (30M model).

Overview
========
We extend the classic Chinchilla scaling law to account for additional tokens D'
produced by a given synthetic data strategy (e.g., repeating, paraphrasing):

    L = E' + B / (D + η · D')^α

where:
    D   = fresh (unique) tokens available
    D'  = additional tokens from the strategy = (epochs - 1) · D
    η   = discount factor: how valuable each D' token is relative to fresh
    E', B, α = classic Chinchilla parameters (fitted on 1-epoch data)

The key idea is that η is strategy-specific.  For a given strategy, we fit a
parametric η(D, D') and can then predict effective token counts at any scale.
Comparing η across strategies (repeat, paraphrase, …) quantifies their
relative value.  See fit_30m_paraphrase.py for the paraphrase strategy.

Fitting procedure (3 steps)
===========================
Step 1  Fit classic scaling law on 1-epoch data
        L = E' + B / D^α   →   E', B, α  (3 params)
        Exclude very small data scales (< 0.5× Chinchilla) which may be in a
        different regime.

Step 2a Back out per-point η
        For each multi-epoch run, solve analytically:
            η = ((B / (L - E'))^{1/α} - D) / D'
        One η per data point — useful for diagnostics.

Step 2b Fit parametric η(D, D')
        Functional form (Michaelis-Menten saturation):
            η = c · (D/N)^{-γ} / (1 + β · D'/D)
        where c = overall scale, γ = data-richness decay, β = saturation rate.
        Key property: η·D' saturates at c·(D/N)^{-γ}·D/β  as D'→∞, giving a
        maximum effective contribution from additional tokens.
        Fit is end-to-end on loss via scipy.optimize.least_squares with grid
        search over initial points.

Step 3  Visualize
        Figure 1 (chinchilla_fits.pdf) — 3 panels:
            Left:   Classic 1-epoch fit  (L vs D)
            Middle: Per-point η collapse (all points on one curve)
            Right:  Parametric η collapse (fit quality)
        Figure 2 (eta_analysis.pdf) — 3 panels:
            Left:   η vs D  (colored by epochs)
            Middle: η vs D' (colored by data scale)
            Right:  Effective tokens D+η·D' vs total tokens D+D'
                    (shows sublinear scaling / saturation)

For this script the strategy is "repeat": D' = (epochs - 1) · D.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit, least_squares, minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dclm_30m import ALL_DATASETS

# ── Constants ──────────────────────────────────────────────────────────
N = 30e6  # model parameters
TTP_RATIO = 20  # Chinchilla token-to-param ratio

FONT_LABEL = 18
FONT_TICK = 16
FONT_LEGEND = 13
FONT_TITLE = 20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Font setup (Palatino Linotype) ─────────────────────────────────────
_FONT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "fonts")
for _fname in ("palatinolinotype_roman.ttf",
               "palatinolinotype_italic.ttf",
               "palatinolinotype_bold.ttf",
               "palatinolinotype_bolditalic.ttf"):
    _fpath = os.path.join(_FONT_DIR, _fname)
    if os.path.exists(_fpath):
        fm.fontManager.addfont(_fpath)
_PALATINO_NAME = fm.FontProperties(
    fname=os.path.join(_FONT_DIR, "palatinolinotype_roman.ttf")
).get_name()
plt.rcParams["font.family"] = _PALATINO_NAME
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def fmt_tokens(x, pos=None):
    """Format token counts as human-readable strings (e.g., 30M, 1.2B)."""
    if x >= 1e9:
        return f"{x/1e9:.1f}B" if x % 1e9 else f"{x/1e9:.0f}B"
    elif x >= 1e6:
        return f"{x/1e6:.0f}M" if x >= 10e6 else f"{x/1e6:.1f}M"
    elif x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Classic scaling law  L = E' + B / D^α
# ═══════════════════════════════════════════════════════════════════════

def scaling_law(D, E_prime, B, alpha):
    """Classic Chinchilla power law."""
    return E_prime + B / D**alpha


def fit_classic(D, loss, E_prime=None):
    """Fit L = E' + B / D^α.

    Args:
        D: array of fresh token counts
        loss: array of validation losses
        E_prime: if given, fix E' and fit only B, α
    Returns:
        (E_prime, B, alpha)
    """
    if E_prime is not None:
        def scaling_fixed(D, B, alpha):
            return E_prime + B / D**alpha
        popt, _ = curve_fit(
            scaling_fixed, D, loss,
            p0=[1e5, 0.5],
            bounds=([0, 0], [np.inf, 2.0]),
            maxfev=10000,
        )
        B, alpha = popt
    else:
        popt, _ = curve_fit(
            scaling_law, D, loss,
            p0=[3.0, 1e5, 0.5],
            bounds=([0, 0, 0], [np.min(loss) - 0.01, np.inf, 2.0]),
            maxfev=10000,
        )
        E_prime, B, alpha = popt

    pred = scaling_law(D, E_prime, B, alpha)
    ss_res = np.sum((loss - pred) ** 2)
    ss_tot = np.sum((loss - np.mean(loss)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"Classic fit: E' = {E_prime:.4f}, B = {B:.2f}, α = {alpha:.4f}, R² = {r2:.6f}")
    return E_prime, B, alpha


def huber_loss(residuals, delta=0.01):
    """Huber (hinge) loss: L2 for |r| < delta, L1 for |r| >= delta."""
    abs_r = np.abs(residuals)
    return np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))


def fit_classic_huber_lbfgs(D, loss, E_prime=None, delta=0.01):
    """Fit L = E' + B / D^α using Huber loss + L-BFGS-B.

    Args:
        D: array of fresh token counts
        loss: array of validation losses
        E_prime: if given, fix E' and fit only B, α
        delta: Huber loss threshold (hinge point)
    Returns:
        (E_prime, B, alpha)
    """
    def objective(params):
        if E_prime is not None:
            B_, alpha_ = params
            Ep_ = E_prime
        else:
            Ep_, B_, alpha_ = params
        pred = Ep_ + B_ / D**alpha_
        return np.sum(huber_loss(loss - pred, delta))

    best_result = None
    best_cost = np.inf
    E_upper = np.min(loss) - 0.01

    Ep_grid = [E_prime] if E_prime is not None else np.linspace(2.5, E_upper - 0.01, 5)
    for Ep0 in Ep_grid:
        for logB in [4, 5, 6, 7, 9]:
            for a0 in [0.3, 0.5, 0.7, 1.0]:
                B0 = 10**logB
                if E_prime is not None:
                    x0 = [B0, a0]
                    bounds = [(1e-6, 1e15), (1e-6, 2.0)]
                else:
                    x0 = [Ep0, B0, a0]
                    bounds = [(1e-6, E_upper), (1e-6, 1e15), (1e-6, 2.0)]
                try:
                    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 50000, "ftol": 1e-15, "gtol": 1e-12})
                    if res.fun < best_cost:
                        best_cost = res.fun
                        best_result = res
                except Exception:
                    pass

    # Refine best L-BFGS result with Nelder-Mead (gradient-free, better for non-smooth Huber)
    if best_result is not None:
        res2 = minimize(objective, best_result.x, method="Nelder-Mead",
                        options={"maxiter": 100000, "xatol": 1e-12, "fatol": 1e-18})
        if res2.fun < best_cost:
            best_cost = res2.fun
            best_result = res2

    if E_prime is not None:
        B, alpha = best_result.x
    else:
        E_prime, B, alpha = best_result.x

    pred = scaling_law(D, E_prime, B, alpha)
    ss_res = np.sum((loss - pred) ** 2)
    ss_tot = np.sum((loss - np.mean(loss)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"Huber+L-BFGS fit (δ={delta}): E' = {E_prime:.4f}, B = {B:.2f}, α = {alpha:.4f}, R² = {r2:.6f}")
    return E_prime, B, alpha


# ═══════════════════════════════════════════════════════════════════════
# STEP 2a: Back out per-point η
# ═══════════════════════════════════════════════════════════════════════

def compute_eta(E_prime, B, alpha, D, D_prime, loss):
    """Solve η analytically for each data point.

    From L = E' + B / (D + η·D')^α:
        η = ((B / (L - E'))^(1/α) - D) / D'
    """
    denom = loss - E_prime
    valid = denom > 0
    eta = np.full_like(D, np.nan)
    eta[valid] = ((B / denom[valid]) ** (1.0 / alpha) - D[valid]) / D_prime[valid]
    return eta


# ═══════════════════════════════════════════════════════════════════════
# STEP 2b: Fit parametric η(D, D')
# ═══════════════════════════════════════════════════════════════════════

def eta_parametric(D, D_prime, c, gamma, beta):
    """η = c · (D/N)^(-γ) / (1 + β · D'/D).

    Michaelis-Menten saturation: η·D' → c·(D/N)^(-γ)·D/β as D'→∞.
    """
    return c * (D / N) ** (-gamma) / (1.0 + beta * (D_prime / D))


def fit_parametric_eta(D, D_prime, loss, E_prime, B, alpha):
    """Fit η params end-to-end by minimizing loss prediction error.

    Args:
        D, D_prime, loss: arrays for multi-epoch (or augmented) data points
        E_prime, B, alpha: fixed classic fit parameters
    Returns:
        (c, gamma, beta)
    """
    def predict_loss(params):
        c, gamma, beta = params
        eta = eta_parametric(D, D_prime, c, gamma, beta)
        D_eff = D + eta * D_prime
        return E_prime + B / D_eff**alpha

    def residuals(params):
        return loss - predict_loss(params)

    best_result = None
    best_cost = np.inf
    for c0 in [0.5, 2.0, 5.0, 10.0]:
        for g0 in [0.2, 0.5, 0.8]:
            for b0 in [0.01, 0.05, 0.2, 1.0]:
                try:
                    res = least_squares(
                        residuals, x0=[c0, g0, b0],
                        bounds=([0, 0, 0], [50, 2, 10]),
                        max_nfev=50000,
                    )
                    if res.cost < best_cost:
                        best_cost = res.cost
                        best_result = res
                except Exception:
                    pass
    c, gamma, beta = best_result.x

    pred = predict_loss(best_result.x)
    ss_res = np.sum((loss - pred) ** 2)
    ss_tot = np.sum((loss - np.mean(loss)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"Parametric η: c={c:.4f}, γ={gamma:.4f}, β={beta:.4f}, R²(loss)={r2:.6f}")
    print(f"  η = {c:.2f}·(D/N)^(-{gamma:.2f}) / (1 + {beta:.2f}·D'/D)")
    return c, gamma, beta


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_chinchilla_fits(ax_classic, ax_perpoint, ax_parametric,
                         D_1ep, loss_1ep, scale_1ep,
                         D_multi, Dp_multi, loss_multi, scale_multi,
                         eta_perpoint, eta_param,
                         E_prime, B, alpha, c, gamma, beta,
                         min_scale=0.5):
    """Populate 3 axes: classic fit, per-point η collapse, parametric η collapse."""
    scales_all = sorted(set(np.concatenate([scale_1ep, scale_multi])))
    cmap = plt.cm.magma_r
    cnorm = plt.Normalize(vmin=np.log2(min(scales_all)) - 1,
                          vmax=np.log2(max(scales_all)) + 1)
    excluded = scale_1ep < min_scale

    # ── Left: classic 1-epoch ──
    ax_classic.scatter(D_1ep[excluded], loss_1ep[excluded], s=80, color="lightgray",
                       edgecolors="k", linewidths=0.5, zorder=5, label="Excluded")
    ax_classic.scatter(D_1ep[~excluded], loss_1ep[~excluded], s=80, color="tab:blue",
                       edgecolors="k", linewidths=0.5, zorder=5, label="1-epoch data")
    D_smooth = np.geomspace(D_1ep.min() * 0.5, D_1ep.max() * 2, 200)
    ax_classic.plot(D_smooth, scaling_law(D_smooth, E_prime, B, alpha), "-",
                    color="tab:red", linewidth=2,
                    label=f"E'={E_prime:.2f}, B={B:.0f}, α={alpha:.3f}")
    ax_classic.axhline(E_prime, color="gray", linestyle="--", alpha=0.5)
    ax_classic.set_xscale("log", base=2)
    ax_classic.set_ylim(E_prime - 0.2, np.max(loss_1ep) + 0.5)
    ax_classic.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_classic.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax_classic.set_title("Classic: L = E' + B / D^α", fontsize=FONT_TITLE - 2)
    ax_classic.tick_params(labelsize=FONT_TICK)
    ax_classic.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_classic.legend(fontsize=FONT_LEGEND)
    ax_classic.grid(alpha=0.3)

    # ── Helper for middle/right panels ──
    def _scatter_collapse(ax, eta_vals, title, subtitle):
        for scale in scales_all:
            m = scale_1ep == scale
            if np.any(m):
                ax.scatter(D_1ep[m], loss_1ep[m], s=50,
                           color=cmap(cnorm(np.log2(scale))),
                           edgecolors="k", linewidths=0.5, zorder=5,
                           label=f"{scale}x")
        valid = ~np.isnan(eta_vals)
        Deff = D_multi[valid] + eta_vals[valid] * Dp_multi[valid]
        for scale in sorted(set(scale_multi[valid])):
            m = scale_multi[valid] == scale
            ax.scatter(Deff[m], loss_multi[valid][m], s=50,
                       color=cmap(cnorm(np.log2(scale))),
                       edgecolors="k", linewidths=0.5, zorder=5)
        all_Deff = np.concatenate([D_1ep, Deff[Deff > 0]])
        Ds = np.geomspace(all_Deff.min() * 0.5, all_Deff.max() * 2, 200)
        ax.plot(Ds, scaling_law(Ds, E_prime, B, alpha), "-", color="tab:red", linewidth=2)
        ax.axhline(E_prime, color="gray", linestyle="--", alpha=0.5)
        all_loss = np.concatenate([loss_1ep, loss_multi[valid]])
        ax.set_ylim(E_prime - 0.2, np.max(all_loss) + 0.5)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Effective tokens  D + η·D'", fontsize=FONT_LABEL)
        ax.set_ylabel("Validation loss", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE - 2)
        ax.text(0.5, 0.02, subtitle, transform=ax.transAxes,
                fontsize=FONT_LEGEND, ha="center", style="italic", color="gray")
        ax.tick_params(labelsize=FONT_TICK)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
        ax.grid(alpha=0.3)

    if ax_perpoint is not None:
        _scatter_collapse(ax_perpoint, eta_perpoint,
                          "Per-point η (exact)", "η solved from each data point")
    _scatter_collapse(ax_parametric, eta_param,
                      "Parametric η (fitted)",
                      f"η = {c:.2f}·(D/N)^(−{gamma:.2f}) / (1+{beta:.2f}·D'/D)")
    ax_parametric.legend(fontsize=FONT_LEGEND - 2, loc="upper right")


def plot_eta_panels(ax_vs_D, ax_vs_Dp,
                    D_multi, Dp_multi, scale_multi, epochs_multi,
                    eta_perpoint, c, gamma, beta):
    """Populate 2 axes: η vs D and η vs D'."""
    valid = ~np.isnan(eta_perpoint)
    D = D_multi[valid]
    Dp = Dp_multi[valid]
    ep = epochs_multi[valid]
    eta_v = eta_perpoint[valid]
    sc = scale_multi[valid]

    unique_epochs = sorted(set(ep))
    cmap_ep = plt.cm.viridis
    norm_ep = plt.Normalize(vmin=np.log2(min(unique_epochs)),
                            vmax=np.log2(max(unique_epochs)))

    # ── Left: η vs D ──
    sc1 = ax_vs_D.scatter(D, eta_v, c=np.log2(ep), cmap=cmap_ep, norm=norm_ep,
                          s=60, edgecolors="k", linewidths=0.5, zorder=5)
    D_smooth = np.geomspace(D.min() * 0.8, D.max() * 1.2, 100)
    for e in unique_epochs:
        Dp_s = (e - 1) * D_smooth
        eta_pred = eta_parametric(D_smooth, Dp_s, c, gamma, beta)
        color = cmap_ep(norm_ep(np.log2(e)))
        label = f"{int(e)} ep" if e in {2, 4, 16, 64, 128} else None
        ax_vs_D.plot(D_smooth, eta_pred, "-", color=color, linewidth=2, alpha=0.7,
                     label=label)
    cbar = plt.colorbar(sc1, ax=ax_vs_D)
    cbar.set_label("log₂(epochs)", fontsize=FONT_LABEL - 2)
    cbar.ax.tick_params(labelsize=FONT_TICK - 2)
    ax_vs_D.set_xscale("log", base=2)
    ax_vs_D.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_vs_D.set_ylabel("η (discount factor)", fontsize=FONT_LABEL)
    ax_vs_D.set_title("η vs D (fresh data)", fontsize=FONT_TITLE)
    ax_vs_D.tick_params(labelsize=FONT_TICK)
    ax_vs_D.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_vs_D.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax_vs_D.grid(alpha=0.3)

    # ── Right: η vs D' ──
    scales = sorted(set(sc))
    cmap_sc = plt.cm.magma_r
    norm_sc = plt.Normalize(vmin=np.log2(min(scales)) - 1,
                            vmax=np.log2(max(scales)) + 1)
    for scale in scales:
        m = sc == scale
        D_val = D[m][0]
        color = cmap_sc(norm_sc(np.log2(scale)))
        ax_vs_Dp.scatter(Dp[m], eta_v[m], s=60, color=color,
                         edgecolors="k", linewidths=0.5, zorder=5, label=f"{scale}x")
        Dp_s = np.geomspace(D_val * 0.8, Dp[m].max() * 1.2, 100)
        ax_vs_Dp.plot(Dp_s, eta_parametric(D_val, Dp_s, c, gamma, beta),
                      "-", color=color, linewidth=2, alpha=0.7)
    ax_vs_Dp.set_xscale("log", base=2)
    ax_vs_Dp.set_xlabel("Additional tokens D'", fontsize=FONT_LABEL)
    ax_vs_Dp.set_ylabel("η (discount factor)", fontsize=FONT_LABEL)
    ax_vs_Dp.set_title("η vs D' (additional data)", fontsize=FONT_TITLE)
    ax_vs_Dp.tick_params(labelsize=FONT_TICK)
    ax_vs_Dp.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_vs_Dp.legend(fontsize=FONT_LEGEND - 1, loc="upper right")
    ax_vs_Dp.grid(alpha=0.3)


def plot_effective_tokens(ax, D_multi, Dp_multi, scale_multi,
                          eta_perpoint, c, gamma, beta):
    """Plot effective tokens D+η·D' vs total tokens D+D'.

    Shows sublinear scaling: as D' grows, effective tokens saturate.
    The η=1 diagonal is the hypothetical "all fresh data" reference.
    Per-point η dots show the data; parametric curves show the fit.
    """
    valid = ~np.isnan(eta_perpoint)
    D_v = D_multi[valid]
    Dp_v = Dp_multi[valid]
    sc_v = scale_multi[valid]
    Deff_pp = D_v + eta_perpoint[valid] * Dp_v

    scales = sorted(set(sc_v))
    cmap_sc = plt.cm.magma_r
    norm_sc = plt.Normalize(vmin=np.log2(min(scales)) - 1,
                            vmax=np.log2(max(scales)) + 1)

    for scale in scales:
        m = sc_v == scale
        D_val = D_v[m][0]
        color = cmap_sc(norm_sc(np.log2(scale)))
        total_tokens = D_val + Dp_v[m]
        # Per-point η data (circles)
        ax.scatter(total_tokens, Deff_pp[m], s=60, color=color,
                   edgecolors="k", linewidths=0.5, zorder=5, label=f"{scale}x")
        # Parametric η prediction (x markers)
        Dp_vals = Dp_v[m]
        eta_pred = eta_parametric(D_val, Dp_vals, c, gamma, beta)
        Deff_pred = D_val + eta_pred * Dp_vals
        ax.scatter(total_tokens, Deff_pred, s=40,
                   color=color, marker="x", linewidths=1.5, zorder=6, alpha=0.8)
        # Parametric curve
        Dp_s = np.geomspace(D_val * 0.5, Dp_v[m].max() * 1.5, 100)
        eta_curve = eta_parametric(D_val, Dp_s, c, gamma, beta)
        Deff_curve = D_val + eta_curve * Dp_s
        ax.plot(D_val + Dp_s, Deff_curve, "-", color=color, linewidth=2, alpha=0.7)

    # Legend entries for marker styles
    ax.scatter([], [], s=60, color="gray", edgecolors="k", linewidths=0.5,
               label="Per-point η")
    ax.scatter([], [], s=40, color="gray", marker="x", linewidths=1.5,
               label="Parametric η")

    # x=y line
    all_total = D_v + Dp_v
    xy_range = np.geomspace(min(D_v) * 0.5, max(all_total) * 1.5, 100)
    ax.plot(xy_range, xy_range, "--", color="gray", linewidth=1.5, alpha=0.5,
            label="η=1 (fresh)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.set_xlabel("Total tokens trained  D + D'  (i.e., FLOPs)", fontsize=FONT_LABEL)
    ax.set_ylabel("Effective tokens  D + η·D'", fontsize=FONT_LABEL)
    ax.set_title("Effective vs Total Tokens", fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND - 1, loc="upper left")
    ax.grid(alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════
# DATA EXTRACTION (specific to multi-epoch repeat data)
# ═══════════════════════════════════════════════════════════════════════

def extract_repeat_data():
    """Extract multi-epoch data from ALL_DATASETS.

    Returns dict with keys: scale, D, epochs, D_prime, loss.
    D' = (epochs - 1) * D for repeated data.
    """
    records = []
    for ds in ALL_DATASETS:
        scale = ds["chinchilla_scale"][0]
        D = scale * TTP_RATIO * N
        for i, ep in enumerate(ds["epochs"]):
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
    data = extract_repeat_data()
    ep1 = data["epochs"] == 1
    multi = data["epochs"] > 1
    fit_mask = ep1 & (data["scale"] >= 0.5)  # exclude data-starved scales

    D_fit, loss_fit = data["D"][fit_mask], data["loss"][fit_mask]

    # ── Step 1: Classic fit (least squares) ──
    print("=" * 60)
    print("METHOD 1: Least-squares (original)")
    print("=" * 60)
    E_prime, B, alpha = fit_classic(D_fit, loss_fit)

    # ── Step 1 alt: Huber + L-BFGS fit ──
    print("\n" + "=" * 60)
    print("METHOD 2: Huber loss + L-BFGS-B")
    print("=" * 60)
    for delta in [0.005, 0.01, 0.02, 0.05]:
        E_prime_h, B_h, alpha_h = fit_classic_huber_lbfgs(D_fit, loss_fit, delta=delta)

    # Use the default delta=0.01 for downstream comparison
    E_prime_h, B_h, alpha_h = fit_classic_huber_lbfgs(D_fit, loss_fit, delta=0.01)

    # ── Compare predictions on 1-epoch data ──
    print("\n" + "=" * 60)
    print("COMPARISON on 1-epoch data")
    print("=" * 60)
    pred_ls = scaling_law(D_fit, E_prime, B, alpha)
    pred_hub = scaling_law(D_fit, E_prime_h, B_h, alpha_h)
    print(f"  {'D':>12s}  {'actual':>8s}  {'LS pred':>8s}  {'Huber pred':>10s}  {'LS err':>8s}  {'Hub err':>8s}")
    for i in range(len(D_fit)):
        print(f"  {D_fit[i]:12.0f}  {loss_fit[i]:8.4f}  {pred_ls[i]:8.4f}  {pred_hub[i]:10.4f}  "
              f"{loss_fit[i]-pred_ls[i]:+8.4f}  {loss_fit[i]-pred_hub[i]:+8.4f}")
    print(f"\n  LS  max|err|={np.max(np.abs(loss_fit - pred_ls)):.5f}  "
          f"RMSE={np.sqrt(np.mean((loss_fit - pred_ls)**2)):.5f}")
    print(f"  Hub max|err|={np.max(np.abs(loss_fit - pred_hub)):.5f}  "
          f"RMSE={np.sqrt(np.mean((loss_fit - pred_hub)**2)):.5f}")

    # ── Use LS fit for downstream η analysis (unchanged) ──
    print("\n" + "=" * 60)
    print("Downstream η analysis (using LS fit)")
    print("=" * 60)

    # ── Step 2a: Per-point η ──
    eta_pp = compute_eta(E_prime, B, alpha,
                         data["D"][multi], data["D_prime"][multi], data["loss"][multi])
    print(f"\nPer-point η: median={np.nanmedian(eta_pp):.4f}, "
          f"mean={np.nanmean(eta_pp):.4f}, "
          f"min={np.nanmin(eta_pp):.4f}, max={np.nanmax(eta_pp):.4f}")

    # ── Step 2b: Parametric η ──
    c, gamma, beta = fit_parametric_eta(
        data["D"][multi], data["D_prime"][multi], data["loss"][multi],
        E_prime, B, alpha,
    )
    eta_param = eta_parametric(data["D"][multi], data["D_prime"][multi], c, gamma, beta)

    # ── Step 3: Plots ──
    # Figure 1: Chinchilla fits (2 panels: classic + parametric collapse)
    fig1, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 7))
    plot_chinchilla_fits(
        ax1, None, ax3,
        data["D"][ep1], data["loss"][ep1], data["scale"][ep1],
        data["D"][multi], data["D_prime"][multi], data["loss"][multi], data["scale"][multi],
        eta_pp, eta_param,
        E_prime, B, alpha, c, gamma, beta,
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(SCRIPT_DIR, "chinchilla_fits.pdf"), bbox_inches="tight")
    fig1.savefig(os.path.join(SCRIPT_DIR, "chinchilla_fits.png"), bbox_inches="tight", dpi=150)
    print(f"Saved chinchilla_fits.pdf")
    plt.close(fig1)

    # Figure 2: Eta analysis (3 panels)
    fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(27, 7))
    plot_eta_panels(
        ax4, ax5,
        data["D"][multi], data["D_prime"][multi],
        data["scale"][multi], data["epochs"][multi],
        eta_pp, c, gamma, beta,
    )
    plot_effective_tokens(
        ax6,
        data["D"][multi], data["D_prime"][multi], data["scale"][multi],
        eta_pp, c, gamma, beta,
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(SCRIPT_DIR, "eta_analysis.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(SCRIPT_DIR, "eta_analysis.png"), bbox_inches="tight", dpi=150)
    print(f"Saved eta_analysis.pdf")
    plt.close(fig2)
