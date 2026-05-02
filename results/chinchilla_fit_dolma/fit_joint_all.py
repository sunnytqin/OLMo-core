"""
One-step joint fit: 1-epoch Chinchilla parameters AND η parameters
estimated *together* on the pooled (1-epoch + multi-epoch) data.

Functional form:

    L(N, D, D') = E + A/N^α + B / (D + η(D, D'; N) · D')^β

For 1-epoch points D' = 0 → η has no effect → reduces to the standard
Chinchilla form  L = E + A/N^α + B/D^β.  For multi-epoch points the
full form is used.

We fit all parameters simultaneously by minimizing Huber on log L,
using the same LSE machinery (LBFGS + grid search + warm start).

Two-stage fit (current default elsewhere) gives a different answer in
principle: it commits to anchors that minimize 1-ep residuals, then
fits η on what's left.  Joint fit lets the multi-epoch data influence
(E, A, B, α, β) directly, so the same residuals are split optimally
between the two stages.

Run with the Muennighoff η form by default; override `ETA_FORM`.
"""

import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))

from data import (OVERFIT_EXCLUDE, SIZES, TTP_RATIO,  # noqa: E402
                  extract_1epoch, extract_multi_epoch, load)
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.1
FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 15, 12, 11, 17


# ──────────────────────────────────────────────────────────────────────
# η forms (supplied as part of the joint forward)
# ──────────────────────────────────────────────────────────────────────

def _eta_muennighoff(p, D, Dp, N):
    """η = R*(1 − exp(−x/R*))/x,  x = D'/D,  R* = R0·(D/N)^ρ."""
    R = p["R0"] * (D / N) ** p["rho"]
    x = Dp / D.clamp(min=1.0)
    safe_x = torch.where(Dp > 0, x, torch.ones_like(x))
    eta = R * (1.0 - torch.exp(-safe_x / R)) / safe_x
    # Where Dp == 0 (1-epoch) η is irrelevant; return 0 so D + η·D' = D.
    return torch.where(Dp > 0, eta, torch.zeros_like(eta))


def _eta_exp_DoverN(p, D, Dp, N):
    """η = η0 · exp(−(D'/D) / (R0·(D/N)^ρ))."""
    R = p["R0"] * (D / N) ** p["rho"]
    eta = p["eta0"] * torch.exp(-(Dp / D.clamp(min=1.0)) / R)
    return torch.where(Dp > 0, eta, torch.zeros_like(eta))


def _eta_muennighoff_RstarN(p, D, Dp, N):
    """exp-sat with explicit R*(D/N, N):
        log R* = log_K + ρ·log(D/N) + σ·log N
        η = R*(1 − e^{−x/R*})/x"""
    log_R = (p["log_K"]
             + p["rho"] * torch.log(D / N)
             + p["sigma"] * torch.log(N))
    R = torch.exp(log_R)
    x = Dp / D.clamp(min=1.0)
    safe_x = torch.where(Dp > 0, x, torch.ones_like(x))
    eta = R * (1.0 - torch.exp(-safe_x / R)) / safe_x
    return torch.where(Dp > 0, eta, torch.zeros_like(eta))


ETA_FORMS: Dict[str, dict] = {
    "Muennighoff": dict(
        fn=_eta_muennighoff,
        eta_param_grid={
            "R0":  [50.0, 500.0],
            "rho": [-1.0, 0.0],
        },
    ),
    "exp(D'/D), R(D/N)": dict(
        fn=_eta_exp_DoverN,
        eta_param_grid={
            "eta0": [0.7, 1.0],
            "R0":   [50.0, 500.0],
            "rho":  [-1.0, 0.0],
        },
    ),
    "Muennighoff R*(N)": dict(
        fn=_eta_muennighoff_RstarN,
        eta_param_grid={
            "log_K": [10.0, 18.0],
            "rho":   [-1.0, 0.0],
            "sigma": [-1.0, 0.0],
        },
    ),
}

# 1-epoch portion of the grid (Hoffmann-style) — kept compact since the
# joint product blows up quickly
CHINCHILLA_GRID = {
    "e":     [0.5, 1.5],
    "a":     [0.0, 10.0],
    "b":     [5.0, 15.0],
    "alpha": [0.2, 0.5],
    "beta":  [0.3, 0.5],
}


def build_grid(eta_param_grid):
    return expand_grid({**CHINCHILLA_GRID, **eta_param_grid})


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def collect_pooled():
    """Return all (1-ep + multi-ep) points pooled across sizes.

    Returns dict with arrays: tags, N, D, Dp (=0 for 1-ep), L, is_multi.
    """
    tags, Ns, Ds, Dps, Ls, is_multi = [], [], [], [], [], []
    for size in SIZES:
        N, datasets = load(size)
        s1, D1, L1 = extract_1epoch(datasets, N, scale_min=0.0)
        for s, d, l in zip(s1, D1, L1):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(0.0)
            Ls.append(l); is_multi.append(False)
        sm, Dm, em, Dpm, Lm, _L1m = extract_multi_epoch(
            datasets, N, scale_min=0.0,
            exclude_overfit=OVERFIT_EXCLUDE.get(size, set()))
        for s, d, dp, l in zip(sm, Dm, Dpm, Lm):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
            Ls.append(l); is_multi.append(True)
    return dict(tags=np.array(tags),
                N=np.array(Ns, dtype=np.float64),
                D=np.array(Ds, dtype=np.float64),
                Dp=np.array(Dps, dtype=np.float64),
                L=np.array(Ls, dtype=np.float64),
                is_multi=np.array(is_multi))


# ──────────────────────────────────────────────────────────────────────
# Joint forward
# ──────────────────────────────────────────────────────────────────────

def make_joint_forward(N_arr, D_arr, Dp_arr, eta_fn):
    log_N = torch.tensor(np.log(N_arr), dtype=torch.float64)
    D_t = torch.tensor(D_arr, dtype=torch.float64)
    Dp_t = torch.tensor(Dp_arr, dtype=torch.float64)
    N_t = torch.tensor(N_arr, dtype=torch.float64)

    def forward(p):
        eta = eta_fn(p, D_t, Dp_t, N_t)              # = 0 where Dp=0
        D_eff = D_t + eta * Dp_t                       # = D where Dp=0
        log_D_eff = torch.log(D_eff)
        terms = torch.stack([
            p["e"].expand_as(log_N),
            p["a"] - p["alpha"] * log_N,
            p["b"] - p["beta"] * log_D_eff,
        ], dim=0)
        return logsumexp_stable(terms, dim=0)

    return forward


def fit_joint_all_sweep(form_name="Muennighoff", delta=DELTA,
                         k_values=(0, 5, 10, 15, 20, 25)):
    """Fit (E, A, B, α, β, η-params) jointly on pooled data, with iterative
    greedy residual drop. Uses one grid search at the start; all subsequent
    fits warm-start from the previous fit's params (single-init).

    Returns dict mapping k → fit result.
    """
    data = collect_pooled()
    form = ETA_FORMS[form_name]
    forward_full = make_joint_forward(data["N"], data["D"], data["Dp"], form["fn"])
    init_grid_full = build_grid(form["eta_param_grid"])

    def fit_on(keep_mask, init_grid):
        log_L = torch.tensor(np.log(data["L"][keep_mask]), dtype=torch.float64)
        forward = make_joint_forward(
            data["N"][keep_mask], data["D"][keep_mask],
            data["Dp"][keep_mask], form["fn"])
        return fit_lse(forward, log_L, init_grid, delta=delta, verbose=False)

    keep = np.ones(len(data["L"]), dtype=bool)
    cumulative_dropped = []
    results = {}
    k_values_sorted = sorted(set(k_values))
    last_params = None

    for k in k_values_sorted:
        # Greedy: extend cumulative_dropped to k entries
        while len(cumulative_dropped) < k:
            init_grid = ([{kk: float(vv) for kk, vv in last_params.items()}]
                         if last_params is not None else init_grid_full)
            res_tmp = fit_on(keep, init_grid)
            with torch.no_grad():
                p_t = {kk: torch.tensor(v, dtype=torch.float64)
                       for kk, v in res_tmp["params"].items()}
                pred_full = forward_full(p_t).numpy()
            resid = np.log(data["L"]) - pred_full
            abs_r = np.abs(resid)
            abs_r[~keep] = -1.0
            worst = int(np.argmax(abs_r))
            cumulative_dropped.append(worst)
            keep[worst] = False
            last_params = res_tmp["params"]

        # Final fit at this k (warm-start from cumulative state)
        init_grid = ([{kk: float(vv) for kk, vv in last_params.items()}]
                     if last_params is not None else init_grid_full)
        res = fit_on(keep, init_grid)
        last_params = res["params"]
        p = res["params"]

        # Build pred + residuals on full data
        p_t = {kk: torch.tensor(v, dtype=torch.float64) for kk, v in p.items()}
        with torch.no_grad():
            pred_full = forward_full(p_t).numpy()
        log_L_full = np.log(data["L"])
        resid_full = log_L_full - pred_full
        rmse_full = float(np.sqrt(np.mean(resid_full ** 2)))
        rmse_kept = float(np.sqrt(np.mean(resid_full[keep] ** 2)))
        m1 = (~data["is_multi"]) & keep
        m2 = data["is_multi"] & keep
        rmse_1ep = float(np.sqrt(np.mean(resid_full[m1] ** 2))) if m1.any() else float("nan")
        rmse_multi = float(np.sqrt(np.mean(resid_full[m2] ** 2))) if m2.any() else float("nan")

        chinch = {
            "E": float(np.exp(p["e"])), "A": float(np.exp(p["a"])),
            "B": float(np.exp(p["b"])),
            "alpha": float(p["alpha"]), "beta": float(p["beta"]),
        }
        eta_par = {kk: float(v) for kk, v in p.items()
                   if kk not in ("e", "a", "b", "alpha", "beta")}
        results[k] = dict(
            chinchilla=chinch, eta_params=eta_par, raw_params=p,
            rmse_full=rmse_full, rmse_kept=rmse_kept,
            rmse_1ep=rmse_1ep, rmse_multi=rmse_multi,
            keep=keep.copy(), dropped=list(cumulative_dropped),
            data=data, pred_full=pred_full, resid=resid_full,
            form_name=form_name,
        )

    return results


def fit_joint_all(form_name="Muennighoff", delta=DELTA, drop_top_k=0):
    """Single-k convenience wrapper around fit_joint_all_sweep."""
    res = fit_joint_all_sweep(form_name, delta, k_values=(drop_top_k,))
    return res[drop_top_k]


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_joint_all(res, path):
    data = res["data"]
    sizes = sorted(set(data["tags"].tolist()),
                   key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_data, ax_res, ax_pari = axes

    for i, size in enumerate(sizes):
        m = data["tags"] == size
        c = cmap(cnorm(i))
        # 1-epoch
        m1 = m & ~data["is_multi"]
        ax_data.scatter(data["D"][m1], data["L"][m1], s=70, color=c,
                        edgecolors="k", linewidths=0.4, marker="o",
                        label=f"{size} 1ep" if i < 2 else None)
        # multi-epoch
        mm = m & data["is_multi"]
        ax_data.scatter(data["D"][mm] + data["Dp"][mm],  # plot vs total tokens
                        data["L"][mm], s=40, color=c,
                        edgecolors="k", linewidths=0.2, marker="x",
                        alpha=0.55)
        # residuals
        ax_res.scatter(data["D"][m], res["resid"][m], s=50, color=c,
                       edgecolors="k", linewidths=0.3,
                       label=size if i % 2 == 0 else None)
        # parity
        ax_pari.scatter(np.exp(res["pred_full"][m]), data["L"][m], s=50,
                        color=c, edgecolors="k", linewidths=0.3,
                        label=size if i % 2 == 0 else None)
        # mark dropped
        dropped_mask = np.zeros(len(data["L"]), dtype=bool)
        dropped_mask[res["dropped"]] = True
        ax_data.scatter(
            data["D"][dropped_mask & m] + data["Dp"][dropped_mask & m],
            data["L"][dropped_mask & m], s=130, facecolors="none",
            edgecolors="red", linewidths=1.5, zorder=10)

    ax_data.set_xscale("log", base=2)
    ax_data.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_data.set_xlabel("D (1ep) or D+D' (multi-ep)", fontsize=FONT_LABEL)
    ax_data.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax_data.set_title(f"Joint fit ({res['form_name']})\n"
                      f"E={res['chinchilla']['E']:.2f}  "
                      f"β={res['chinchilla']['beta']:.3f}  "
                      f"RMSE={res['rmse_kept']:.3f}",
                      fontsize=FONT_TITLE)
    ax_data.tick_params(labelsize=FONT_TICK)
    ax_data.grid(alpha=0.3)
    ax_data.legend(fontsize=FONT_LEGEND, loc="upper right")

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_res.set_xlabel("D", fontsize=FONT_LABEL)
    ax_res.set_ylabel("residual (log L)", fontsize=FONT_LABEL)
    ax_res.set_title("Residuals", fontsize=FONT_TITLE)
    ax_res.tick_params(labelsize=FONT_TICK)
    ax_res.grid(alpha=0.3)

    lo = min(data["L"].min(), np.exp(res["pred_full"]).min()) * 0.95
    hi = max(data["L"].max(), np.exp(res["pred_full"]).max()) * 1.05
    ax_pari.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6)
    ax_pari.set_xlabel("Predicted L", fontsize=FONT_LABEL)
    ax_pari.set_ylabel("Observed L", fontsize=FONT_LABEL)
    ax_pari.set_title("Parity plot", fontsize=FONT_TITLE)
    ax_pari.tick_params(labelsize=FONT_TICK)
    ax_pari.grid(alpha=0.3)

    fig.suptitle("One-step joint Chinchilla + η fit  "
                 "(red circles = points dropped by residual)",
                 fontsize=FONT_TITLE + 1, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 92)
    print("ONE-STEP joint fit: (E, A, B, α, β) + η params, all together")
    print("=" * 92)
    K_VALUES = (0, 5, 10, 15, 20, 25, 30)
    for form in ["Muennighoff", "exp(D'/D), R(D/N)"]:
        print(f"\n--- form: {form} ---")
        sweep = fit_joint_all_sweep(form_name=form, delta=DELTA, k_values=K_VALUES)
        for k in sorted(sweep.keys()):
            r = sweep[k]
            ch = r["chinchilla"]
            ep = r["eta_params"]
            print(f"  k={k:>2d}  E={ch['E']:.3f}  A={ch['A']:.1f}  "
                  f"B={ch['B']:.0f}  α={ch['alpha']:.3f}  β={ch['beta']:.3f}  |  "
                  + "  ".join(f"{kk}={vv:.3g}" for kk, vv in ep.items())
                  + f"  |  RMSE 1ep={r['rmse_1ep']:.3f}  multi={r['rmse_multi']:.3f}")

        # Pick canonical k where β stops moving
        ks = sorted(sweep.keys())
        betas_seq = [sweep[k]["chinchilla"]["beta"] for k in ks]
        canonical = ks[-1]
        for i in range(1, len(betas_seq)):
            if abs(betas_seq[i] - betas_seq[i - 1]) < 0.01:
                canonical = ks[i]
                break
        print(f"  canonical k (|Δβ|<0.01): {canonical}")

        tag = form.replace(" ", "_").replace("/", "over").replace("(", "").replace(")", "").replace(",", "")
        plot_joint_all(
            sweep[canonical],
            path=os.path.join(SCRIPT_DIR, f"fit_joint_all_{tag}.pdf"))


if __name__ == "__main__":
    main()
