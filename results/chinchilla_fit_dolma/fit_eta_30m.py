"""
η fitting for multi-epoch Dolma-30M training.

Setup
-----
Fix the 1-epoch Chinchilla fit (E, B, β) and model multi-epoch loss as
    L = E + B / (D + η · D')^β
where D' = (epochs − 1) · D is the extra tokens from repetition.  η is
the effective-token multiplier for repeated data.  Physical sanity check:
    η < 1  (each repeated token is worth less than a fresh token).

The functional form of η is not known.  We fit several candidates and
compare by log-L RMSE, LOO-CV RMSE (honest OOB), residual structure,
and per-scale fit quality.

Candidate forms (simplest to most expressive):
  const           : η = c
  power(D/N)      : η = c · (D/N)^(−γ)
  power(D'/D)     : η = c · (D'/D)^(−γ)
  sat(D'/D)       : η = c / (1 + b · D'/D)                   [Michaelis-Menten]
  sat × (D/N)     : η = c · (D/N)^(−γ) / (1 + b · D'/D)
  double power    : η = c · (D/N)^(−γ₁) · (D'/D)^(−γ₂)

"sat" = saturating in D'/D (a.k.a. Michaelis-Menten): grows linearly at
small D'/D, flattens at large D'/D.  η·D' → c·D/b as D'→∞, giving a
ceiling on the effective-token gain from repetition.

Data
----
We use only scales ≥ 0.5x (the same k=3 cut used for the 1-epoch fit).
Keeping the small scales here contaminates η: the 1-epoch fit doesn't
describe 0.05x/0.1x/0.25x, so the multi-epoch η at those scales absorbs
the 1-epoch misfit and comes out > 1 (non-physical).

Fit procedure: LSE + Huber(δ=1e-3) + L-BFGS with grid-search init.
Over-fit multi-epoch points (u-shape in loss vs epochs) are excluded.
"""

import os
import sys
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dolma_30m import ALL_DATASETS  # noqa: E402
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402
from fit_1epoch_30m import N, TTP_RATIO, classic_law, fmt_tokens  # noqa: E402

# ── 1-epoch anchors (k=3 cut, δ=0.1 from fit_1epoch_30m.py) ──
# We switched from δ=1e-3 (E=2.537, β=0.355) to δ=0.1 because the shallower
# β=0.355 extrapolates poorly past the 1-epoch fit range, producing η > 1
# (non-physical) for low-epoch multi-epoch points.  With β=0.486 all per-point
# η end up < 1, as expected.  See writeup.md for details.
E_FIT = 3.010
B_FIT = 47022.47
BETA_FIT = 0.4863

DELTA = 1e-3  # Huber loss threshold (Besiroglu-style, L1-ish on log-L)
SCALE_MIN = 0.5  # drop multi-epoch data below this (matches 1-epoch k=3 cut)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# u-shape / overfit points to exclude up front (from dolma_30m scan):
OVERFIT_EXCLUDE = {(0.05, 128), (0.1, 128)}

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 16, 13, 11, 18


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def extract_multi_epoch(datasets, scale_min=SCALE_MIN, exclude_overfit=True):
    """Return (scale, D, epochs, D', L) for multi-epoch (e>1) points at
    scale >= scale_min, optionally excluding u-shape overfits."""
    rows = []
    for ds in datasets:
        scale = ds["chinchilla_scale"][0]
        if scale < scale_min:
            continue
        D = scale * TTP_RATIO * N
        for i, ep in enumerate(ds["epochs"]):
            if ep == 1:
                continue
            loss = ds["validation_loss"][i]
            if np.isnan(loss):
                continue
            if exclude_overfit and (scale, ep) in OVERFIT_EXCLUDE:
                continue
            rows.append((scale, D, ep, (ep - 1) * D, loss))
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def per_point_eta(D, Dp, L, E=E_FIT, B=B_FIT, beta=BETA_FIT):
    """Solve η analytically at each data point:  η = ((B/(L−E))^(1/β) − D) / D'."""
    denom = L - E
    eta = np.full_like(D, np.nan)
    v = denom > 0
    eta[v] = ((B / denom[v]) ** (1.0 / beta) - D[v]) / Dp[v]
    return eta


# ──────────────────────────────────────────────────────────────────────
# Candidate η forms
# ──────────────────────────────────────────────────────────────────────
# Each form takes (params_dict, D, Dp, N) with D, Dp, N as torch tensors
# and returns η of shape (R,).

def _eta_const(p, D, Dp, N):
    return p["c"] * torch.ones_like(D)


def _eta_power_DoverN(p, D, Dp, N):
    return p["c"] * (D / N) ** (-p["gamma"])


def _eta_power_DpoverD(p, D, Dp, N):
    return p["c"] * (Dp / D) ** (-p["gamma"])


def _eta_mm_DpoverD(p, D, Dp, N):
    return p["c"] / (1.0 + p["b"] * (Dp / D))


def _eta_mm_withDoverN(p, D, Dp, N):
    return p["c"] * (D / N) ** (-p["gamma"]) / (1.0 + p["b"] * (Dp / D))


def _eta_double_power(p, D, Dp, N):
    return p["c"] * (D / N) ** (-p["gamma1"]) * (Dp / D) ** (-p["gamma2"])


FORMS: Dict[str, dict] = {
    "const": dict(
        fn=_eta_const,
        grid=expand_grid({"c": [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]}),
        desc="η = c",
    ),
    "power(D/N)": dict(
        fn=_eta_power_DoverN,
        grid=expand_grid({
            "c":     [0.1, 0.5, 1.0, 5.0, 20.0],
            "gamma": [0.0, 0.2, 0.5, 1.0, 1.5],
        }),
        desc="η = c · (D/N)^(−γ)",
    ),
    "power(D'/D)": dict(
        fn=_eta_power_DpoverD,
        grid=expand_grid({
            "c":     [0.1, 0.3, 0.7, 1.5, 3.0],
            "gamma": [-0.3, 0.0, 0.3, 0.7, 1.0],
        }),
        desc="η = c · (D'/D)^(−γ)",
    ),
    "sat(D'/D)": dict(
        fn=_eta_mm_DpoverD,
        grid=expand_grid({
            "c": [0.3, 1.0, 3.0, 10.0, 30.0],
            "b": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        }),
        desc="η = c / (1 + b · D'/D)",
    ),
    "sat × (D/N)": dict(
        fn=_eta_mm_withDoverN,
        grid=expand_grid({
            "c":     [0.5, 2.0, 10.0, 30.0],
            "gamma": [0.0, 0.2, 0.5, 1.0],
            "b":     [0.003, 0.01, 0.1, 1.0],
        }),
        desc="η = c · (D/N)^(−γ) / (1 + b · D'/D)",
    ),
    "double power": dict(
        fn=_eta_double_power,
        grid=expand_grid({
            "c":      [0.1, 0.5, 2.0, 10.0],
            "gamma1": [-0.2, 0.0, 0.3, 0.7],
            "gamma2": [-0.2, 0.0, 0.3, 0.7],
        }),
        desc="η = c · (D/N)^(−γ₁) · (D'/D)^(−γ₂)",
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Fit machinery
# ──────────────────────────────────────────────────────────────────────

def _make_forward(eta_fn: Callable, D, Dp,
                  E=E_FIT, B=B_FIT, beta=BETA_FIT):
    """Build a forward_fn: params → log-L pred, using η(params, D, Dp, N)."""
    D_t = torch.tensor(D, dtype=torch.float64)
    Dp_t = torch.tensor(Dp, dtype=torch.float64)
    N_t = torch.tensor(float(N), dtype=torch.float64)
    e_c = torch.tensor(float(np.log(E)), dtype=torch.float64)
    b_c = torch.tensor(float(np.log(B)), dtype=torch.float64)
    beta_c = float(beta)

    def forward(p):
        eta = eta_fn(p, D_t, Dp_t, N_t)
        D_eff = D_t + eta * Dp_t
        terms = torch.stack(
            [e_c.expand_as(D_t), b_c - beta_c * torch.log(D_eff)], dim=0
        )
        return logsumexp_stable(terms, dim=0)

    return forward


def fit_form(form: dict, D, Dp, L,
             E=E_FIT, B=B_FIT, beta=BETA_FIT, delta=DELTA) -> dict:
    """Fit one η form; return params, per-point η, residuals, fit metrics."""
    log_L = torch.tensor(np.log(L), dtype=torch.float64)
    forward = _make_forward(form["fn"], D, Dp, E, B, beta)
    res = fit_lse(forward, log_L, form["grid"], delta=delta, verbose=False)

    # Per-point η under fitted params
    p_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in res["params"].items()}
    D_t = torch.tensor(D, dtype=torch.float64)
    Dp_t = torch.tensor(Dp, dtype=torch.float64)
    N_t = torch.tensor(float(N), dtype=torch.float64)
    with torch.no_grad():
        eta_pred = form["fn"](p_t, D_t, Dp_t, N_t).numpy()
        log_L_pred = forward(p_t).numpy()
    resid = np.log(L) - log_L_pred
    return dict(
        params=res["params"],
        eta_pred=eta_pred,
        log_L_pred=log_L_pred,
        resid=resid,
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        max_abs=float(np.max(np.abs(resid))),
        r2=1.0 - np.sum(resid ** 2) / np.sum((np.log(L) - np.log(L).mean()) ** 2),
    )


def leave_one_out(form: dict, D, Dp, L,
                  E=E_FIT, B=B_FIT, beta=BETA_FIT, delta=DELTA) -> np.ndarray:
    """Per-point LOO residual: fit on others, predict at held-out i."""
    n = len(D)
    oob = np.zeros(n)
    for i in range(n):
        keep = np.ones(n, dtype=bool); keep[i] = False
        res = fit_form(form, D[keep], Dp[keep], L[keep], E, B, beta, delta)
        # Predict at the held-out point
        p_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in res["params"].items()}
        D_t = torch.tensor(D[i:i+1], dtype=torch.float64)
        Dp_t = torch.tensor(Dp[i:i+1], dtype=torch.float64)
        N_t = torch.tensor(float(N), dtype=torch.float64)
        with torch.no_grad():
            eta_i = form["fn"](p_t, D_t, Dp_t, N_t).item()
        pred_i = classic_law(D[i], E, B, beta * 0 + beta) if False else \
                 E + B / (D[i] + eta_i * Dp[i]) ** beta
        oob[i] = np.log(L[i]) - np.log(pred_i)
    return oob


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

def plot_diagnostics(results, best_name, D, Dp, scale, ep, L, eta_pp, path):
    """3-panel figure for the best form:
        (1) η vs D' per scale  (points = per-point η, curves = parametric)
        (2) Loss collapse: L vs D_eff = D + η·D'  +  1-epoch fit curve
        (3) Residuals (pred − obs) vs D', per scale
    """
    res = results[best_name]
    scales = sorted(set(scale.tolist()))
    cmap = plt.cm.magma_r
    cnorm = plt.Normalize(vmin=np.log2(min(scales)) - 1,
                          vmax=np.log2(max(scales)) + 1)

    def color_of(s):
        return cmap(cnorm(np.log2(s)))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    ax_eta, ax_coll, ax_resid = axes
    fig.suptitle(f"η diagnostics  |  best form: {best_name}   "
                 f"({FORMS[best_name]['desc']})",
                 fontsize=FONT_TITLE + 1, y=1.01)

    # ── (1) η vs D' per scale ──
    for s in scales:
        m = scale == s
        c = color_of(s)
        # per-point η (solved from data)
        valid = ~np.isnan(eta_pp)
        ax_eta.scatter(Dp[m & valid], eta_pp[m & valid], s=60, color=c,
                       edgecolors="k", linewidths=0.3, zorder=5,
                       label=f"{s:g}x")
        # parametric η (from fitted form) — a smooth curve per scale
        D_s = s * TTP_RATIO * N
        Dp_smooth = np.geomspace(max(Dp[m].min() * 0.5, D_s * 0.5),
                                 Dp[m].max() * 1.5, 80)
        p_t = {k: torch.tensor(v, dtype=torch.float64)
               for k, v in res["params"].items()}
        D_arr = torch.full_like(torch.tensor(Dp_smooth), D_s, dtype=torch.float64)
        Dp_arr = torch.tensor(Dp_smooth, dtype=torch.float64)
        N_t = torch.tensor(float(N), dtype=torch.float64)
        with torch.no_grad():
            eta_curve = FORMS[best_name]["fn"](p_t, D_arr, Dp_arr, N_t).numpy()
        ax_eta.plot(Dp_smooth, eta_curve, "-", color=c, linewidth=1.8, alpha=0.8)
    ax_eta.set_xscale("log", base=2)
    ax_eta.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_eta.set_xlabel("Additional tokens D'", fontsize=FONT_LABEL)
    ax_eta.set_ylabel("η  (effective-token multiplier)", fontsize=FONT_LABEL)
    ax_eta.set_title("η vs D'  (points = per-point, curves = parametric)",
                     fontsize=FONT_TITLE)
    ax_eta.tick_params(labelsize=FONT_TICK)
    ax_eta.legend(fontsize=FONT_LEGEND - 1, loc="best", title="scale",
                  ncol=2, title_fontsize=FONT_LEGEND)
    ax_eta.grid(alpha=0.3, which="both")

    # ── (2) Loss collapse: L vs D + η·D' ──
    D_eff = D + res["eta_pred"] * Dp
    for s in scales:
        m = scale == s
        c = color_of(s)
        ax_coll.scatter(D_eff[m], L[m], s=60, color=c,
                        edgecolors="k", linewidths=0.3, zorder=5,
                        label=f"{s:g}x")
    # 1-epoch fit curve
    D_smooth = np.geomspace(D_eff.min() * 0.5, D_eff.max() * 2, 300)
    L_smooth = classic_law(D_smooth, E_FIT, B_FIT, BETA_FIT)
    ax_coll.plot(D_smooth, L_smooth, "-", color="tab:red", linewidth=2,
                 label=f"1-ep fit: E={E_FIT:.2f}, B={B_FIT:.0f}, β={BETA_FIT:.3f}")
    ax_coll.axhline(E_FIT, color="gray", linestyle=":", alpha=0.4)
    ax_coll.set_xscale("log", base=2)
    ax_coll.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_coll.set_xlabel("Effective tokens  D + η·D'", fontsize=FONT_LABEL)
    ax_coll.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax_coll.set_title("Loss collapse onto 1-epoch curve", fontsize=FONT_TITLE)
    ax_coll.tick_params(labelsize=FONT_TICK)
    ax_coll.legend(fontsize=FONT_LEGEND - 1)
    ax_coll.grid(alpha=0.3)

    # ── (3) Residuals ──
    for s in scales:
        m = scale == s
        c = color_of(s)
        ax_resid.scatter(Dp[m], res["resid"][m], s=60, color=c,
                         edgecolors="k", linewidths=0.3, zorder=5,
                         label=f"{s:g}x")
    ax_resid.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_resid.set_xscale("log", base=2)
    ax_resid.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_resid.set_xlabel("Additional tokens D'", fontsize=FONT_LABEL)
    ax_resid.set_ylabel("log-loss residual  (obs − pred)", fontsize=FONT_LABEL)
    ax_resid.set_title("Residuals vs D' by scale", fontsize=FONT_TITLE)
    ax_resid.tick_params(labelsize=FONT_TICK)
    ax_resid.legend(fontsize=FONT_LEGEND - 1, loc="best")
    ax_resid.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_form_comparison(results, D, Dp, scale, L, eta_pp, path):
    """Grid of 2 rows × N forms.
        Row 1: η vs D' per scale  (points = per-point; curves = fitted form).
        Row 2: log-L residuals vs D' per scale.
    Panels are ordered by LOO RMSE (best → worst).
    """
    forms_sorted = sorted(results.keys(), key=lambda k: results[k]["loo_rmse"])
    scales = sorted(set(scale.tolist()))
    cmap = plt.cm.magma_r
    cnorm = plt.Normalize(vmin=np.log2(min(scales)) - 1,
                          vmax=np.log2(max(scales)) + 1)

    def color_of(s):
        return cmap(cnorm(np.log2(s)))

    n_forms = len(forms_sorted)
    fig, axes = plt.subplots(2, n_forms, figsize=(4.2 * n_forms, 9),
                             squeeze=False)

    # Shared y-axis ranges so panels are directly comparable
    eta_all = np.concatenate([results[n]["eta_pred"] for n in forms_sorted] +
                             [eta_pp[~np.isnan(eta_pp)]])
    eta_ymin = max(1e-2, np.nanmin(eta_all) * 0.6)
    eta_ymax = np.nanmax(eta_all) * 1.5
    resid_all = np.concatenate([results[n]["resid"] for n in forms_sorted])
    resid_lim = np.max(np.abs(resid_all)) * 1.2

    for col, name in enumerate(forms_sorted):
        res = results[name]
        ax_eta, ax_r = axes[0, col], axes[1, col]

        # η vs D' (points = per-point, curves = parametric)
        for s in scales:
            m = (scale == s)
            c = color_of(s)
            v = ~np.isnan(eta_pp) & m
            ax_eta.scatter(Dp[v], eta_pp[v], s=45, color=c,
                           edgecolors="k", linewidths=0.3, zorder=5,
                           label=f"{s:g}x" if col == 0 else None)
            # parametric curve at this scale
            D_s = s * TTP_RATIO * N
            Dp_range = np.geomspace(Dp[m].min() * 0.7, Dp[m].max() * 1.3, 80)
            p_t = {k: torch.tensor(v_, dtype=torch.float64)
                   for k, v_ in res["params"].items()}
            D_arr = torch.full_like(torch.tensor(Dp_range), D_s, dtype=torch.float64)
            Dp_arr = torch.tensor(Dp_range, dtype=torch.float64)
            N_t = torch.tensor(float(N), dtype=torch.float64)
            with torch.no_grad():
                curve = FORMS[name]["fn"](p_t, D_arr, Dp_arr, N_t).numpy()
            ax_eta.plot(Dp_range, curve, "-", color=c, linewidth=1.6, alpha=0.85)
        ax_eta.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax_eta.set_xscale("log", base=2)
        ax_eta.set_yscale("log")
        ax_eta.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
        ax_eta.set_ylim(eta_ymin, eta_ymax)
        title = (f"{name}\nRMSE={res['rmse']:.3f}  "
                 f"LOO={res['loo_rmse']:.3f}  R²={res['r2']:.3f}")
        ax_eta.set_title(title, fontsize=FONT_LEGEND + 1)
        if col == 0:
            ax_eta.set_ylabel("η  (points + fit)", fontsize=FONT_LABEL)
            ax_eta.legend(fontsize=FONT_LEGEND - 2, loc="best",
                          title="scale", ncol=2)
        ax_eta.tick_params(labelsize=FONT_TICK - 2)
        ax_eta.grid(alpha=0.3, which="both")

        # log-L residuals vs D'
        for s in scales:
            m = (scale == s)
            c = color_of(s)
            ax_r.scatter(Dp[m], res["resid"][m], s=45, color=c,
                         edgecolors="k", linewidths=0.3, zorder=5)
        ax_r.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax_r.set_xscale("log", base=2)
        ax_r.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
        ax_r.set_ylim(-resid_lim, resid_lim)
        ax_r.set_xlabel("D'", fontsize=FONT_LABEL - 2)
        if col == 0:
            ax_r.set_ylabel("residual  (obs − pred log L)",
                            fontsize=FONT_LABEL)
        ax_r.tick_params(labelsize=FONT_TICK - 2)
        ax_r.grid(alpha=0.3)

    fig.suptitle("η form comparison  (columns sorted by LOO RMSE, best→worst)",
                 fontsize=FONT_TITLE + 2, y=1.00)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    scale, D, ep, Dp, L = extract_multi_epoch(ALL_DATASETS, scale_min=SCALE_MIN,
                                               exclude_overfit=True)
    print(f"\n{'='*100}")
    print(f"η fit on Dolma-30M multi-epoch data")
    print(f"{'='*100}")
    print(f"  1-epoch anchors (held fixed):  E={E_FIT:.3f}  B={B_FIT:.2f}  β={BETA_FIT:.4f}")
    print(f"  Points: {len(D)} multi-epoch  "
          f"(scale >= {SCALE_MIN}x;  overfit excluded: {sorted(OVERFIT_EXCLUDE)})")
    print(f"  Scales: {sorted(set(scale.tolist()))}")

    # Per-point η (diagnostic) + sanity check
    eta_pp = per_point_eta(D, Dp, L)
    print(f"  Per-point η:  min={np.nanmin(eta_pp):.3f}  "
          f"median={np.nanmedian(eta_pp):.3f}  max={np.nanmax(eta_pp):.3f}  "
          f"(n_valid={(~np.isnan(eta_pp)).sum()})")
    n_gt1 = int(np.sum(eta_pp > 1.0))
    if n_gt1:
        print(f"  ⚠ {n_gt1}/{len(eta_pp)} points have η > 1 "
              f"(repeated token worth > fresh — non-physical)")
    else:
        print(f"  ✓ all η < 1  (fresh tokens always more valuable than repeats)")

    # Fit each candidate form
    results = {}
    print(f"\n{'-'*100}")
    print(f"{'form':<16s}  {'n_par':>5}  {'RMSE(logL)':>10}  "
          f"{'max|Δlog|':>10}  {'R²(logL)':>9}  {'LOO RMSE':>10}  params")
    print(f"{'-'*100}")

    for name, form in FORMS.items():
        res = fit_form(form, D, Dp, L)
        loo = leave_one_out(form, D, Dp, L)
        res["loo_rmse"] = float(np.sqrt(np.mean(loo ** 2)))
        res["loo_resid"] = loo
        results[name] = res
        pstr = "  ".join(f"{k}={v:.4g}" for k, v in res["params"].items())
        print(f"{name:<16s}  {len(res['params']):>5d}  {res['rmse']:>10.4f}  "
              f"{res['max_abs']:>10.4f}  {res['r2']:>9.4f}  "
              f"{res['loo_rmse']:>10.4f}  {pstr}")

    # Rank by LOO
    print(f"\n{'='*100}")
    print("Ranking by LOO RMSE (honest OOB):")
    for name, res in sorted(results.items(), key=lambda kv: kv[1]["loo_rmse"]):
        print(f"  {name:<16s}  LOO RMSE = {res['loo_rmse']:.4f}")

    best_name = min(results, key=lambda k: results[k]["loo_rmse"])
    print(f"\nBest form:  {best_name}   ({FORMS[best_name]['desc']})")

    # Plots
    tag = best_name.replace(" ", "_").replace("/", "over").replace("(", "").replace(")", "")
    plot_diagnostics(results, best_name, D, Dp, scale, ep, L, eta_pp,
                     path=os.path.join(SCRIPT_DIR, f"fit_eta_30m_{tag}.pdf"))
    plot_form_comparison(results, D, Dp, scale, L, eta_pp,
                         path=os.path.join(SCRIPT_DIR,
                                           "fit_eta_30m_form_comparison.pdf"))
    return results


if __name__ == "__main__":
    main()
