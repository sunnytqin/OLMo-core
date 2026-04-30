"""
η fitting for multi-epoch Dolma runs, across all model sizes.

For each size with multi-epoch data, fit

    L = E_eff(N) + B / (D + η(D, D'; N) · D')^β

using the joint-Chinchilla anchors (E, A, B, α, β) — so E_eff(N) = E + A/N^α
and (B, β) are shared across sizes. This lets η be compared across sizes
on a consistent footing.

Reports per-size η parameters for each candidate form, plus a joint fit
that pools multi-epoch data across all sizes (testing whether a single
(c, γ, b) captures the full N × D × D' dependence).

η forms:
  const           η = c
  power(D/N)      η = c · (D/N)^(−γ)
  power(D'/D)     η = c · (D'/D)^(−γ)
  sat(D'/D)       η = c / (1 + b · D'/D)           [saturating in D'/D]
  sat × (D/N)     η = c · (D/N)^(−γ) / (1 + b · D'/D)      ← best from 30M
  double power    η = c · (D/N)^(−γ₁) · (D'/D)^(−γ₂)

Fit machinery: LSE + Huber(δ=1e-3) + L-BFGS + grid-search init.
"""

import os
import sys
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))

from data import (DEFAULT_SCALE_MIN, OVERFIT_EXCLUDE, SIZES, TTP_RATIO,  # noqa: E402
                  extract_multi_epoch, load)
from fit_chinchilla_joint import (collect_1epoch_all_sizes, fit_joint,  # noqa: E402
                                   predict, topk_residual_drop_sweep)
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 1e-3       # η fit — Besiroglu-style L1-ish Huber
SCALE_MIN = DEFAULT_SCALE_MIN

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 15, 12, 10, 17


# ──────────────────────────────────────────────────────────────────────
# η forms
# ──────────────────────────────────────────────────────────────────────

def _eta_const(p, D, Dp, N):       return p["c"] * torch.ones_like(D)
def _eta_pow_DoverN(p, D, Dp, N):  return p["c"] * (D / N) ** (-p["gamma"])
def _eta_pow_DpoverD(p, D, Dp, N): return p["c"] * (Dp / D) ** (-p["gamma"])
def _eta_sat_DpoverD(p, D, Dp, N): return p["c"] / (1.0 + p["b"] * (Dp / D))

def _eta_sat_mul_DoverN(p, D, Dp, N):
    return p["c"] * (D / N) ** (-p["gamma"]) / (1.0 + p["b"] * (Dp / D))

def _eta_double_power(p, D, Dp, N):
    return p["c"] * (D / N) ** (-p["gamma1"]) * (Dp / D) ** (-p["gamma2"])


# N-dependent saturation: b_eff = b0 · (N/N_ref)^κ.  Captures the observed
# trend that larger models saturate faster (b rises with N per-size fits).
_N_REF = 30e6


def _eta_sat_Nb(p, D, Dp, N):
    N_ref = torch.tensor(_N_REF, dtype=N.dtype)
    b_eff = p["b0"] * (N / N_ref) ** p["kappa"]
    return p["c"] * (D / N) ** (-p["gamma"]) / (1.0 + b_eff * (Dp / D))


# ── Exponential decay forms (Muennighoff-inspired) ─────────────────────
# At D'/D → 0 the multiplier approaches η₀; saturation scale R sets where
# η drops to η₀/e.  If η₀ ≤ 1, η ≤ 1 everywhere automatically.

def _eta_exp(p, D, Dp, N):
    """η = η₀ · exp(−(D'/D) / R)."""
    return p["eta0"] * torch.exp(-(Dp / D) / p["R"])


def _eta_exp_DoverN(p, D, Dp, N):
    """η = η₀ · exp(−(D'/D) / R(D/N)),  R = R₀ · (D/N)^ρ."""
    R = p["R0"] * (D / N) ** p["rho"]
    return p["eta0"] * torch.exp(-(Dp / D) / R)


def _eta_muennighoff(p, D, Dp, N):
    """Muennighoff '23 Eq 5 (exact form, derived from D'_eff = U_D + U_D · R*·(1 − e^{−R/R*})):
        η = R*·(1 − exp(−x/R*)) / x,   x = D'/D,   R* = R₀ · (D/N)^ρ.
    Algebraically forces η→1 as x→0 (first repeated token = fresh) and
    η → 0 as x → ∞ (full saturation)."""
    R = p["R0"] * (D / N) ** p["rho"]
    x = Dp / D
    return R * (1.0 - torch.exp(-x / R)) / x


FORMS: Dict[str, dict] = {
    "const": dict(
        fn=_eta_const,
        grid=expand_grid({"c": [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5]}),
        desc="η = c",
    ),
    "power(D/N)": dict(
        fn=_eta_pow_DoverN,
        grid=expand_grid({
            "c":     [0.05, 0.2, 0.5, 1.0, 2.5, 5.0, 15.0, 40.0],
            "gamma": [-0.2, 0.0, 0.15, 0.3, 0.5, 0.8, 1.2, 1.6],
        }),
        desc="η = c · (D/N)^(−γ)",
    ),
    "power(D'/D)": dict(
        fn=_eta_pow_DpoverD,
        grid=expand_grid({
            "c":     [0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0],
            "gamma": [-0.3, -0.1, 0.0, 0.2, 0.4, 0.6, 0.9, 1.3],
        }),
        desc="η = c · (D'/D)^(−γ)",
    ),
    "sat(D'/D)": dict(
        fn=_eta_sat_DpoverD,
        grid=expand_grid({
            "c": [0.2, 0.5, 1.0, 2.0, 4.0, 10.0, 30.0, 80.0],
            "b": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        }),
        desc="η = c / (1 + b · D'/D)",
    ),
    "sat × (D/N)": dict(
        fn=_eta_sat_mul_DoverN,
        grid=expand_grid({
            "c":     [0.5, 2.0, 6.0, 15.0, 40.0],
            "gamma": [0.0, 0.2, 0.4, 0.7, 1.1],
            "b":     [0.003, 0.03, 0.1, 0.3, 1.0],
        }),
        desc="η = c · (D/N)^(−γ) / (1 + b · D'/D)",
    ),
    "double power": dict(
        fn=_eta_double_power,
        grid=expand_grid({
            "c":      [0.1, 0.5, 2.0, 8.0, 25.0],
            "gamma1": [-0.2, 0.0, 0.3, 0.6, 1.0],
            "gamma2": [-0.2, 0.0, 0.3, 0.6, 1.0],
        }),
        desc="η = c · (D/N)^(−γ₁) · (D'/D)^(−γ₂)",
    ),
    "sat × (D/N), b(N)": dict(
        fn=_eta_sat_Nb,
        grid=expand_grid({
            "c":     [0.5, 2.0, 6.0, 15.0, 40.0],
            "gamma": [0.0, 0.2, 0.4, 0.7, 1.1],
            "b0":    [0.01, 0.05, 0.2, 1.0],
            "kappa": [0.0, 0.5, 1.0, 1.5],
        }),
        desc=f"η = c · (D/N)^(−γ) / (1 + b₀·(N/{_N_REF:.0e})^κ · D'/D)",
    ),
    "exp(D'/D)": dict(
        fn=_eta_exp,
        grid=expand_grid({
            "eta0": [0.3, 0.5, 0.7, 1.0],
            "R":    [0.5, 2.0, 5.0, 20.0, 100.0],
        }),
        desc="η = η₀ · exp(−(D'/D) / R)",
    ),
    "exp(D'/D), R(D/N)": dict(
        fn=_eta_exp_DoverN,
        grid=expand_grid({
            "eta0": [0.3, 0.6, 1.0],
            "R0":   [1.0, 5.0, 20.0, 100.0],
            "rho":  [-1.0, -0.3, 0.0, 0.5],
        }),
        desc="η = η₀ · exp(−(D'/D) / (R₀·(D/N)^ρ))",
    ),
    "Muennighoff Eq 5": dict(
        fn=_eta_muennighoff,
        grid=expand_grid({
            "R0":  [0.5, 2.0, 5.0, 20.0, 100.0],
            "rho": [-1.0, -0.3, 0.0, 0.5],
        }),
        desc="η = R*·(1−e^(−x/R*))/x  with R* = R₀·(D/N)^ρ ; x=D'/D",
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Fit machinery
# ──────────────────────────────────────────────────────────────────────

def _make_forward(eta_fn, D_np, Dp_np, N_np, E_eff_np, B, beta):
    """Build forward(params) → log L pred.

    E_eff can be per-point (np array) to support pooled fits across sizes.
    """
    D_t    = torch.tensor(D_np, dtype=torch.float64)
    Dp_t   = torch.tensor(Dp_np, dtype=torch.float64)
    N_t    = torch.tensor(N_np, dtype=torch.float64)
    log_E  = torch.tensor(np.log(E_eff_np), dtype=torch.float64)
    log_B  = torch.tensor(float(np.log(B)), dtype=torch.float64)
    beta_c = float(beta)

    def forward(p):
        eta = eta_fn(p, D_t, Dp_t, N_t)
        D_eff = D_t + eta * Dp_t
        terms = torch.stack([log_E, log_B - beta_c * torch.log(D_eff)], dim=0)
        return logsumexp_stable(terms, dim=0)
    return forward


def per_point_eta(D, Dp, L, E_eff, B, beta):
    """Invert L = E_eff + B/D_eff^β to solve η per point.

    LEGACY: depends on E_eff(N), so per-size 1-epoch fit residuals
    propagate into per-point η.  Prefer per_point_eta_diff below.
    """
    denom = L - E_eff
    eta = np.full_like(D, np.nan)
    v = denom > 0
    eta[v] = ((B / denom[v]) ** (1.0 / beta) - D[v]) / Dp[v]
    return eta


def per_point_eta_diff(D, Dp, L, L_1ep, B, beta):
    """Solve η from the *empirical* loss difference between multi-epoch and
    1-epoch at the same (size, scale).

        ΔL = L(1ep) − L(multi-ep) = B/D^β − B/(D + η·D')^β

    so that

        1/D_eff^β = 1/D^β − ΔL/B,   D_eff = (1/D^β − ΔL/B)^(−1/β),
        η = (D_eff − D) / D'.

    Cancels E_eff(N) entirely → cleaner per-point η, free of the
    1-epoch fit's per-size residuals.  Only B and β enter.

    Returns NaN where 1/D^β − ΔL/B ≤ 0 (unphysical: would require
    η·D' < 0 to explain the loss drop, i.e. multi-epoch is too good
    for the assumed β).
    """
    delta_L = L_1ep - L  # > 0 if multi-ep improves over 1-ep
    inv_Db = 1.0 / D ** beta
    inv_Deff_b = inv_Db - delta_L / B
    eta = np.full_like(D, np.nan)
    v = inv_Deff_b > 0
    D_eff = np.full_like(D, np.nan)
    D_eff[v] = inv_Deff_b[v] ** (-1.0 / beta)
    eta[v] = (D_eff[v] - D[v]) / Dp[v]
    return eta


def fit_form(form: dict, D, Dp, N_arr, L, E_eff, B, beta, delta=DELTA):
    """Fit a single form. All arrays are np, length R.
    E_eff and N_arr can be scalar or per-point."""
    if np.isscalar(E_eff):
        E_eff = np.full_like(D, float(E_eff))
    if np.isscalar(N_arr):
        N_arr = np.full_like(D, float(N_arr))
    log_L = torch.tensor(np.log(L), dtype=torch.float64)
    forward = _make_forward(form["fn"], D, Dp, N_arr, E_eff, B, beta)
    res = fit_lse(forward, log_L, form["grid"], delta=delta)
    p_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in res["params"].items()}
    D_t = torch.tensor(D, dtype=torch.float64)
    Dp_t = torch.tensor(Dp, dtype=torch.float64)
    N_t = torch.tensor(N_arr, dtype=torch.float64)
    with torch.no_grad():
        eta_pred = form["fn"](p_t, D_t, Dp_t, N_t).numpy()
        logL_pred = forward(p_t).numpy()
    resid = np.log(L) - logL_pred
    return dict(
        params=res["params"], eta_pred=eta_pred, resid=resid,
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        max_abs=float(np.max(np.abs(resid))),
        r2=1 - np.sum(resid ** 2) / np.sum((np.log(L) - np.log(L).mean()) ** 2),
    )


def leave_one_out(form: dict, D, Dp, N_arr, L, E_eff, B, beta,
                   delta=DELTA, warm_init=None):
    """Warm-started LOO: starts each fold from the full-data fit's params,
    skipping the grid search.  ~50× faster than re-doing the grid each fold."""
    if np.isscalar(E_eff): E_eff = np.full_like(D, float(E_eff))
    if np.isscalar(N_arr): N_arr = np.full_like(D, float(N_arr))
    # Build a single-point "grid" from the warm-start params
    if warm_init is None:
        res_full = fit_form(form, D, Dp, N_arr, L, E_eff, B, beta, delta)
        warm_init = res_full["params"]
    single_grid = [{k: float(v) for k, v in warm_init.items()}]
    form_warm = {**form, "grid": single_grid}

    n = len(D)
    oob = np.full(n, np.nan)
    for i in range(n):
        keep = np.ones(n, dtype=bool); keep[i] = False
        try:
            res = fit_form(form_warm, D[keep], Dp[keep], N_arr[keep], L[keep],
                           E_eff[keep], B, beta, delta)
        except RuntimeError:
            # warm-start failed for this fold; fall back to full grid
            try:
                res = fit_form(form, D[keep], Dp[keep], N_arr[keep], L[keep],
                               E_eff[keep], B, beta, delta)
            except RuntimeError:
                continue
        p_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in res["params"].items()}
        with torch.no_grad():
            eta_i = form["fn"](
                p_t,
                torch.tensor(D[i:i+1], dtype=torch.float64),
                torch.tensor(Dp[i:i+1], dtype=torch.float64),
                torch.tensor(N_arr[i:i+1], dtype=torch.float64),
            ).item()
        pred_i = E_eff[i] + B / (D[i] + eta_i * Dp[i]) ** beta
        oob[i] = np.log(L[i]) - np.log(pred_i)
    return oob


# ──────────────────────────────────────────────────────────────────────
# Data pooling
# ──────────────────────────────────────────────────────────────────────

def collect_multi_epoch_all_sizes(scale_min=SCALE_MIN):
    """Return (size_tag, N, scale, D, epochs, D', L, L_1ep) arrays across
    all sizes with multi-epoch data, with per-size overfit exclusions."""
    tags, Ns, scales, Ds, eps, Dps, Ls, L_1eps = [], [], [], [], [], [], [], []
    for size in SIZES:
        N, datasets = load(size)
        s, D, ep, Dp, L, L_1ep = extract_multi_epoch(
            datasets, N, scale_min=scale_min,
            exclude_overfit=OVERFIT_EXCLUDE.get(size, set()))
        if len(D) == 0:
            continue
        tags.extend([size] * len(D))
        Ns.extend([N] * len(D))
        scales.extend(s)
        Ds.extend(D)
        eps.extend(ep)
        Dps.extend(Dp)
        Ls.extend(L)
        L_1eps.extend(L_1ep)
    return (np.array(tags), np.array(Ns, dtype=np.float64),
            np.array(scales), np.array(Ds), np.array(eps),
            np.array(Dps), np.array(Ls), np.array(L_1eps))


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_per_size_eta(results, path, title):
    """One column per size × one row per form — showing η vs D'/D on log-log.
    Rows are the four top-performing forms, sorted by joint LOO."""
    sizes = sorted(results.keys(), key=lambda t: SIZES[t][0])
    forms = [
        "sat × (D/N)",
        "sat × (D/N), b(N)",
        "exp(D'/D), R(D/N)",
        "Muennighoff Eq 5",
    ]
    fig, axes = plt.subplots(len(forms), len(sizes),
                             figsize=(3.4 * len(sizes), 2.7 * len(forms)),
                             squeeze=False)
    for row, form_name in enumerate(forms):
        for col, size in enumerate(sizes):
            ax = axes[row, col]
            r = results[size].get(form_name)
            if r is None:
                ax.set_visible(False)
                continue
            # scatter η_parametric(data) vs D'/D
            d = results[size]
            ax.scatter(d["Dp"] / d["D"], r["eta_pred"], s=22, color="tab:blue",
                       edgecolors="k", linewidths=0.3, zorder=5)
            # per-point η (target) — grey
            ax.scatter(d["Dp"] / d["D"], d["eta_pp"], s=22, color="gray",
                       edgecolors="k", linewidths=0.2, alpha=0.55, zorder=4)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"{size}  (N={SIZES[size][0]/1e6:.0f}M)",
                             fontsize=FONT_LEGEND)
            if col == 0:
                ax.set_ylabel(f"{form_name}\nη", fontsize=FONT_LEGEND)
            if row == len(forms) - 1:
                ax.set_xlabel("D'/D", fontsize=FONT_LEGEND)
            ax.tick_params(labelsize=FONT_TICK - 2)
            ax.grid(alpha=0.25, which="both")
            if "rmse" in r:
                ax.text(0.03, 0.03, f"LOO={r.get('loo_rmse', r['rmse']):.3f}",
                        transform=ax.transAxes, fontsize=FONT_LEGEND - 2,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.8))
    fig.suptitle(title, fontsize=FONT_TITLE + 1, y=1.00)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_AvsB_per_size(per_size, path):
    """One panel per size; both Muennighoff (A) and sat-x-(D/N)-b(N) (B)
    fitted η curves overlaid on the per-point η scatter."""
    sizes = sorted(per_size.keys(), key=lambda t: SIZES[t][0])
    nrows, ncols = 1, len(sizes)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 4.0),
                             squeeze=False)
    A_name = "Muennighoff Eq 5"
    B_name = "sat × (D/N), b(N)"

    for col, size in enumerate(sizes):
        ax = axes[0, col]
        d = per_size[size]
        N = SIZES[size][0]

        # per-point η
        v = ~np.isnan(d["eta_pp"])
        ax.scatter(d["Dp"][v] / d["D"][v], d["eta_pp"][v], s=35,
                   color="gray", edgecolors="k", linewidths=0.3, alpha=0.7,
                   zorder=4, label="per-point η")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Smooth curve grid for plotting fits — we span the data D'/D range
        # for the typical scale at this size
        scales = sorted(set((d["D"] / (TTP_RATIO * N)).tolist()))
        for j, scale in enumerate(scales):
            D_val = scale * TTP_RATIO * N
            mask = np.isclose(d["D"], D_val)
            if not mask.any():
                continue
            x_min = max(d["Dp"][mask].min() * 0.7, D_val * 0.1) / D_val
            x_max = (d["Dp"][mask].max() * 1.4) / D_val
            x_range = np.geomspace(x_min, x_max, 80)
            Dp_range = x_range * D_val
            D_arr = torch.full((len(x_range),), float(D_val), dtype=torch.float64)
            Dp_arr = torch.tensor(Dp_range, dtype=torch.float64)
            N_t = torch.full_like(D_arr, float(N))

            for form_name, color, ls in [(A_name, "tab:red", "-"),
                                          (B_name, "tab:blue", "--")]:
                if form_name not in d:
                    continue
                p = d[form_name]["params"]
                p_t = {k: torch.tensor(v_, dtype=torch.float64) for k, v_ in p.items()}
                with torch.no_grad():
                    eta_curve = FORMS[form_name]["fn"](p_t, D_arr, Dp_arr, N_t).numpy()
                # Only label on first scale to avoid legend explosion
                lbl = (f"A. Muennighoff" if (form_name == A_name and j == 0)
                       else (f"B. sat × (D/N), b(N)" if (form_name == B_name and j == 0)
                             else None))
                ax.plot(x_range, eta_curve, ls, color=color, linewidth=1.6,
                        alpha=0.85, label=lbl)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("D'/D", fontsize=FONT_LABEL - 1)
        if col == 0:
            ax.set_ylabel("η", fontsize=FONT_LABEL)
        loo_A = d.get(A_name, {}).get("loo_rmse", float("nan"))
        loo_B = d.get(B_name, {}).get("loo_rmse", float("nan"))
        ax.set_title(f"{size}  (N={N/1e6:.0f}M)\n"
                     f"LOO  A={loo_A:.3f}  B={loo_B:.3f}",
                     fontsize=FONT_LEGEND + 1)
        ax.tick_params(labelsize=FONT_TICK - 2)
        ax.grid(alpha=0.3, which="both")
        if col == 0:
            ax.legend(fontsize=FONT_LEGEND - 1, loc="best")

    fig.suptitle("η fit: Form A (Muennighoff) vs Form B (sat × (D/N), b(N))  "
                 "— per-point η as background dots",
                 fontsize=FONT_TITLE, y=1.05)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_joint_AvsB(tags, D, Dp, N_arr, L, eta_pp, results_joint, path):
    """Two-panel figure: form A fit and form B fit, both on the pooled data,
    coloured by size. Lets the reader see where each form deviates."""
    sizes = sorted(set(tags), key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)
    A_name, B_name = "Muennighoff Eq 5", "sat × (D/N), b(N)"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Joint η fit — A vs B, residuals coloured by size",
                 fontsize=FONT_TITLE + 1, y=0.995)

    for col, (form_name, label) in enumerate(
            [(A_name, "A. Muennighoff Eq 5"),
             (B_name, "B. sat × (D/N), b(N)")]):
        if form_name not in results_joint:
            continue
        r = results_joint[form_name]
        ax_eta = axes[0, col]
        ax_res = axes[1, col]

        for i, size in enumerate(sizes):
            m = tags == size
            color = cmap(cnorm(i))
            # per-point η
            valid = m & ~np.isnan(eta_pp)
            ax_eta.scatter(Dp[valid] / D[valid], eta_pp[valid], s=42,
                           color=color, edgecolors="k", linewidths=0.3,
                           alpha=0.55, zorder=4,
                           label=size if col == 0 else None)
            # parametric η at fitted params
            ax_eta.scatter(Dp[m] / D[m], r["eta_pred"][m], s=42,
                           color=color, marker="x", linewidths=1.6, zorder=6)
            ax_res.scatter(Dp[m] / D[m], r["resid"][m], s=50, color=color,
                           edgecolors="k", linewidths=0.3,
                           label=size if col == 0 else None)
        ax_eta.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax_eta.set_xscale("log"); ax_eta.set_yscale("log")
        ax_eta.set_xlabel("D'/D", fontsize=FONT_LABEL)
        ax_eta.set_ylabel("η", fontsize=FONT_LABEL)
        ax_eta.set_title(f"{label}\n(dots = per-point, × = fitted)",
                         fontsize=FONT_TITLE - 1)
        ax_eta.tick_params(labelsize=FONT_TICK)
        if col == 0:
            ax_eta.legend(fontsize=FONT_LEGEND, loc="best", ncol=2)
        ax_eta.grid(alpha=0.3, which="both")

        ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax_res.set_xscale("log")
        ax_res.set_xlabel("D'/D", fontsize=FONT_LABEL)
        ax_res.set_ylabel("log-loss residual (obs − pred)", fontsize=FONT_LABEL)
        ax_res.set_title(f"residuals  (RMSE={r['rmse']:.4f},  LOO={r['loo_rmse']:.4f})",
                         fontsize=FONT_TITLE - 1)
        ax_res.tick_params(labelsize=FONT_TICK)
        if col == 0:
            ax_res.legend(fontsize=FONT_LEGEND, loc="best", ncol=2)
        ax_res.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_joint_eta(tags, D, Dp, N_arr, L, eta_pp, res, form_name, path):
    """For the pooled joint fit, plot η vs D'/D coloured by size, with
    fitted-curve overlays."""
    sizes = sorted(set(tags), key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_eta, ax_res = axes

    for i, size in enumerate(sizes):
        m = tags == size
        color = cmap(cnorm(i))
        ax_eta.scatter((Dp[m] / D[m]), eta_pp[m], s=50, color=color,
                       edgecolors="k", linewidths=0.3, alpha=0.6,
                       label=f"{size}  (per-point)")
        ax_eta.scatter((Dp[m] / D[m]), res["eta_pred"][m], s=40, color=color,
                       marker="x", linewidths=1.3)
        ax_res.scatter(Dp[m] / D[m], res["resid"][m], s=50, color=color,
                       edgecolors="k", linewidths=0.3,
                       label=f"{size}")

    ax_eta.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_eta.set_xscale("log"); ax_eta.set_yscale("log")
    ax_eta.set_xlabel("D'/D", fontsize=FONT_LABEL)
    ax_eta.set_ylabel("η", fontsize=FONT_LABEL)
    ax_eta.set_title(f"Joint η fit — {form_name}\n"
                     f"(circles = per-point; × = fitted)",
                     fontsize=FONT_TITLE)
    ax_eta.legend(fontsize=FONT_LEGEND - 1, loc="best", ncol=2)
    ax_eta.grid(alpha=0.3, which="both")

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("D'/D", fontsize=FONT_LABEL)
    ax_res.set_ylabel("log-loss residual (obs − pred)", fontsize=FONT_LABEL)
    ax_res.set_title(f"Residuals  (RMSE={res['rmse']:.4f})", fontsize=FONT_TITLE)
    ax_res.legend(fontsize=FONT_LEGEND - 1, loc="best")
    ax_res.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def diagnose_size(per_size, joint_results, anchor, size: str):
    """Print per-point η and the joint-fit residuals for one size.
    Useful for understanding η>1 or η<0 points."""
    if size not in per_size:
        print(f"\n(no per-size data for {size})")
        return
    d = per_size[size]
    N = SIZES[size][0]
    E_eff_ = anchor["E"] + anchor["A"] / N ** anchor["alpha"]
    B, beta = anchor["B"], anchor["beta"]

    print(f"\n{'='*100}")
    print(f"Per-point η diagnostic for {size}  (N={N/1e6:.0f}M, E_eff={E_eff_:.3f})")
    print(f"{'='*100}")

    # Predicted 1-epoch L at same D (no repetition)
    L_pred_1ep = E_eff_ + B / d["D"] ** beta
    resid_1ep = np.log(d["L"]) - np.log(L_pred_1ep)  # positive if obs > pred at same D

    # Per-point η (already in per_size)
    eta_pp = d["eta_pp"]

    # Sort by scale then epochs for readable output
    order = np.lexsort((d["ep"], d["D"]))
    hdr_tot = "D+D'"
    print(f"  {'scale':>7s}  {'ep':>4s}  {'D':>10s}  {'D/N':>7s}  "
          f"{hdr_tot:>10s}  {'L_obs':>7s}  {'L_pred':>7s}  {'Δlog':>7s}  "
          f"{'η_pp':>7s}  flag")
    for i in order:
        scale = d["D"][i] / (TTP_RATIO * N)
        dn = d["D"][i] / N
        tot = d["D"][i] + d["Dp"][i]
        flag = []
        if eta_pp[i] > 1:   flag.append("η>1")
        if eta_pp[i] < 0:   flag.append("η<0!")
        flag_s = " ".join(flag)
        print(f"  {scale:>7.2f}  {int(d['ep'][i]):>4d}  {d['D'][i]:>10.2e}  "
              f"{dn:>7.1f}  {tot:>10.2e}  {d['L'][i]:>7.4f}  "
              f"{L_pred_1ep[i]:>7.4f}  {resid_1ep[i]:>+7.3f}  "
              f"{eta_pp[i]:>7.3f}  {flag_s}")

    # Sign pattern of 1-epoch residuals — if systematically positive, joint
    # fit under-predicts for this size.
    mean_res = float(np.mean(resid_1ep))
    n_pos = int(np.sum(resid_1ep > 0))
    print(f"\n  Δlog summary (joint 1-ep pred vs observed multi-ep L at same D):")
    print(f"    mean = {mean_res:+.4f}   positive: {n_pos}/{len(resid_1ep)}")
    if abs(mean_res) > 0.01:
        direction = "over" if mean_res < 0 else "under"
        print(f"    joint Chinchilla {direction}-predicts loss at this N "
              f"→ η inversion is biased")


def main():
    # 1) Joint 5-param Chinchilla anchors via residual-based outlier drop
    print(f"\n{'='*100}")
    print("Fitting joint Chinchilla anchors (E, A, B, α, β) — iterative "
          "residual drop on all 1-epoch points")
    print(f"{'='*100}")
    tags_1ep, N_1ep, _, D_1ep, L_1ep = collect_1epoch_all_sizes(scale_min=0.0)
    drop_sweep, _, _ = topk_residual_drop_sweep(
        tags_1ep, N_1ep, D_1ep, L_1ep,
        k_values=[0, 5, 10, 15, 20, 25], delta=0.1, iterative=True)
    # Pick smallest k where β stabilizes
    k_sorted = sorted(drop_sweep.keys())
    betas_seq = [drop_sweep[k]["p"]["beta"] for k in k_sorted]
    canonical_k = k_sorted[-1]
    for i in range(1, len(betas_seq)):
        if abs(betas_seq[i] - betas_seq[i - 1]) < 0.01:
            canonical_k = k_sorted[i]
            break
    anchor = drop_sweep[canonical_k]["p"]
    E_joint, A_joint = anchor["E"], anchor["A"]
    B_joint, alpha_joint, beta_joint = anchor["B"], anchor["alpha"], anchor["beta"]
    print(f"\nUsing canonical k={canonical_k} anchors:")
    print(f"  E={E_joint:.4f}  A={A_joint:.2f}  B={B_joint:.2f}  "
          f"α={alpha_joint:.4f}  β={beta_joint:.4f}")

    def E_eff(N):
        return E_joint + A_joint / N ** alpha_joint

    # 2) Per-size η fits (for sizes with multi-epoch data).
    #    We keep scale_min=0.5x for the η fit because including small-scale
    #    multi-epoch points pollutes per-point η (the joint anchors miss
    #    those scales the most, even after residual drop on 1-epoch data).
    #    The "no scale cut" change applies to the 1-epoch fit only.
    eta_scale_min = 0.5
    print(f"\n{'='*100}")
    print(f"Per-size η fits  (scale ≥ {eta_scale_min}×, δ={DELTA})")
    print(f"{'='*100}")
    per_size: Dict[str, Dict] = {}
    for size in SIZES:
        N, datasets = load(size)
        s, D, ep, Dp, L, L_1ep = extract_multi_epoch(
            datasets, N, scale_min=eta_scale_min,
            exclude_overfit=OVERFIT_EXCLUDE.get(size, set()))
        if len(D) < 5:
            print(f"  {size}: only {len(D)} multi-epoch points — skipping per-size fit")
            continue
        E_size = E_eff(N)
        # Two per-point η solvers — report both for comparison.
        eta_pp = per_point_eta_diff(D, Dp, L, L_1ep, B_joint, beta_joint)
        eta_pp_legacy = per_point_eta(D, Dp, L, E_size, B_joint, beta_joint)
        n_gt1 = int(np.sum(eta_pp > 1))
        print(f"\n  {size}  (N={N/1e6:.0f}M, n={len(D)}, E_eff={E_size:.3f}):")
        valid = ~np.isnan(eta_pp)
        print(f"    per-point η (ΔL form):    min={np.nanmin(eta_pp):.3f} "
              f"median={np.nanmedian(eta_pp):.3f} max={np.nanmax(eta_pp):.3f} "
              f"(η>1: {n_gt1}/{int(valid.sum())})")
        print(f"    per-point η (E_eff form): min={np.nanmin(eta_pp_legacy):.3f} "
              f"median={np.nanmedian(eta_pp_legacy):.3f} "
              f"max={np.nanmax(eta_pp_legacy):.3f}  (legacy)")
        print(f"    {'form':<16s}  {'RMSE':>7s}  {'LOO':>7s}  {'R²':>6s}  params")
        per_size[size] = {"D": D, "Dp": Dp, "eta_pp": eta_pp,
                          "eta_pp_legacy": eta_pp_legacy,
                          "L": L, "L_1ep": L_1ep, "ep": ep}
        for name, form in FORMS.items():
            res = fit_form(form, D, Dp, N, L, E_size, B_joint, beta_joint)
            loo = leave_one_out(form, D, Dp, N, L, E_size, B_joint, beta_joint,
                                warm_init=res["params"])
            valid = ~np.isnan(loo)
            res["loo_rmse"] = float(np.sqrt(np.mean(loo[valid] ** 2))) if valid.any() else float("nan")
            per_size[size][name] = res
            pstr = "  ".join(f"{k}={v:.3g}" for k, v in res["params"].items())
            print(f"    {name:<16s}  {res['rmse']:>7.4f}  {res['loo_rmse']:>7.4f}  "
                  f"{res['r2']:>6.3f}  {pstr}")

    # 3) Joint η fit across all sizes (pooled)
    print(f"\n{'='*100}")
    print("Joint η fit across all sizes  (pooled multi-epoch data)")
    print(f"{'='*100}")
    tags, Ns, scales, Ds, eps, Dps, Ls, L_1eps = collect_multi_epoch_all_sizes(
        eta_scale_min)
    E_eff_arr = np.array([E_eff(n) for n in Ns])
    eta_pp_pool = per_point_eta_diff(Ds, Dps, Ls, L_1eps, B_joint, beta_joint)
    eta_pp_pool_legacy = per_point_eta(Ds, Dps, Ls, E_eff_arr, B_joint, beta_joint)
    valid_pool = ~np.isnan(eta_pp_pool)
    n_gt1 = int(np.sum(eta_pp_pool[valid_pool] > 1))
    print(f"  pooled n = {len(Ds)} across sizes: "
          f"{sorted(set(tags.tolist()))}")
    print(f"  per-point η (ΔL):    min={np.nanmin(eta_pp_pool):.3f} "
          f"median={np.nanmedian(eta_pp_pool):.3f} "
          f"max={np.nanmax(eta_pp_pool):.3f} "
          f"(η>1: {n_gt1}/{int(valid_pool.sum())})")
    print(f"  per-point η (E_eff): min={np.nanmin(eta_pp_pool_legacy):.3f} "
          f"median={np.nanmedian(eta_pp_pool_legacy):.3f} "
          f"max={np.nanmax(eta_pp_pool_legacy):.3f}  (legacy)")
    print(f"\n  {'form':<16s}  {'RMSE':>7s}  {'LOO':>7s}  {'R²':>6s}  params")

    joint_results: Dict[str, Dict] = {}
    for name, form in FORMS.items():
        res = fit_form(form, Ds, Dps, Ns, Ls, E_eff_arr, B_joint, beta_joint)
        loo = leave_one_out(form, Ds, Dps, Ns, Ls, E_eff_arr, B_joint, beta_joint,
                            warm_init=res["params"])
        res["loo_rmse"] = float(np.sqrt(np.mean(loo ** 2)))
        joint_results[name] = res
        pstr = "  ".join(f"{k}={v:.3g}" for k, v in res["params"].items())
        print(f"  {name:<16s}  {res['rmse']:>7.4f}  {res['loo_rmse']:>7.4f}  "
              f"{res['r2']:>6.3f}  {pstr}")

    best_joint = min(joint_results, key=lambda k: joint_results[k]["loo_rmse"])
    print(f"\nBest joint form: {best_joint}   ({FORMS[best_joint]['desc']})")

    # ── 4) Per-point diagnostic for 190M ──────────────────────────────
    diagnose_size(per_size, joint_results, anchor, "190m")

    # Plots
    plot_per_size_eta(
        per_size, path=os.path.join(SCRIPT_DIR, "fit_eta_per_size.pdf"),
        title=f"η fit per size (δ={DELTA}, scale ≥ {SCALE_MIN}×)")

    plot_joint_eta(tags, Ds, Dps, Ns, Ls, eta_pp_pool,
                   joint_results[best_joint], best_joint,
                   path=os.path.join(SCRIPT_DIR, "fit_eta_joint.pdf"))

    # New: A vs B head-to-head plots
    plot_AvsB_per_size(
        per_size, path=os.path.join(SCRIPT_DIR, "fit_eta_AvsB_per_size.pdf"))

    plot_joint_AvsB(
        tags, Ds, Dps, Ns, Ls, eta_pp_pool, joint_results,
        path=os.path.join(SCRIPT_DIR, "fit_eta_AvsB_joint.pdf"))

    return per_size, joint_results


if __name__ == "__main__":
    main()
