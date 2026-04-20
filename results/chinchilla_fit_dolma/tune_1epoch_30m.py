"""
Tune the 1-epoch Chinchilla fit on Dolma-30M.

Follows the Besiroglu 2024 methodology (see Analyzing_Chinchilla_data.ipynb):
  • Drop the top-k highest-loss points (outliers by loss, not by scale).
  • Fit  log L = LSE(e, b - β log D)  with Huber(δ=1e-3) + L-BFGS.
  • Literature-style log-spaced grid over initial (e, b, β).
  • Warm-started bootstrap (n_boot ≈ 500) from the main fit to get
    parameter SEs and 95% CIs.
  • Leave-one-out (LOO) for out-of-sample goodness-of-fit.
  • Report residuals on held-out (dropped) points for honest OOB check.

Sweeps k ∈ {0, 1, 2, 3} — i.e. drop 0, 1, 2, or 3 outliers — to find the
cut where residuals become structureless and parameters stabilize.
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dolma_30m import ALL_DATASETS  # noqa: E402
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402

N = 30e6
TTP_RATIO = 20
DELTA = 1e-3  # Besiroglu's choice; effectively L1 in our residual range
N_BOOT = 500
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 16, 13, 11, 18

# Log-spaced grid (Besiroglu style, adapted to 3 params since N is fixed)
GRID = expand_grid({
    "e":    [-1.0, 0.0, 0.5, 1.0, 1.5],
    "b":    [2.0, 5.0, 8.0, 11.0, 14.0],
    "beta": [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5],
})


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def extract_1epoch(datasets):
    rows = []
    for ds in datasets:
        scale = ds["chinchilla_scale"][0]
        idx = ds["epochs"].index(1)
        loss = ds["validation_loss"][idx]
        if not np.isnan(loss):
            rows.append((scale, scale * TTP_RATIO * N, loss))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


def classic_law(D, E, B, beta):
    return E + B / D ** beta


# ──────────────────────────────────────────────────────────────────────
# Fit helpers
# ──────────────────────────────────────────────────────────────────────

def _forward_factory(log_D: torch.Tensor):
    def forward(p):
        terms = torch.stack([p["e"].expand_as(log_D),
                             p["b"] - p["beta"] * log_D], dim=0)
        return logsumexp_stable(terms, dim=0)
    return forward


def fit(D, L, init_grid=GRID, delta=DELTA) -> Dict:
    log_D = torch.tensor(np.log(D), dtype=torch.float64)
    log_L = torch.tensor(np.log(L), dtype=torch.float64)
    res = fit_lse(_forward_factory(log_D), log_L, init_grid, delta=delta)
    p = res["params"]
    return {"e": p["e"], "b": p["b"], "beta": p["beta"],
            "E": float(np.exp(p["e"])), "B": float(np.exp(p["b"]))}


def fit_warm(D, L, init, delta=DELTA) -> Dict:
    return fit(D, L, init_grid=[init], delta=delta)


# ──────────────────────────────────────────────────────────────────────
# Uncertainty / OOB
# ──────────────────────────────────────────────────────────────────────

def leave_one_out(D, L, init, delta=DELTA) -> np.ndarray:
    """Return per-point OOB log-residual (log L_i − log pred_i_from_others)."""
    n = len(D)
    oob = np.zeros(n)
    for i in range(n):
        keep = np.ones(n, dtype=bool); keep[i] = False
        p = fit_warm(D[keep], L[keep], init, delta=delta)
        pred_i = classic_law(D[i], p["E"], p["B"], p["beta"])
        oob[i] = np.log(L[i]) - np.log(pred_i)
    return oob


def bootstrap(D, L, init, n_boot=N_BOOT, seed=42, delta=DELTA) -> np.ndarray:
    """Warm-started bootstrap.  Returns array of shape (n_valid, 3) for E,B,β."""
    rng = np.random.default_rng(seed)
    n = len(D)
    rows: List[List[float]] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 3:
            continue
        try:
            p = fit_warm(D[idx], L[idx], init, delta=delta)
            rows.append([p["E"], p["B"], p["beta"]])
        except Exception:
            continue
    return np.array(rows)


def summary_stats(samples: np.ndarray) -> Tuple[np.ndarray, ...]:
    mu = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1)
    ci_lo = np.percentile(samples, 2.5, axis=0)
    ci_hi = np.percentile(samples, 97.5, axis=0)
    return mu, se, ci_lo, ci_hi


# ──────────────────────────────────────────────────────────────────────
# Residual scoring
# ──────────────────────────────────────────────────────────────────────

def resid_log(D, L, p):
    pred = classic_law(D, p["E"], p["B"], p["beta"])
    return np.log(L) - np.log(pred)


def rmse(r: np.ndarray) -> float:
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")


def maxabs(r: np.ndarray) -> float:
    return float(np.max(np.abs(r))) if len(r) else float("nan")


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_diagnostics(results, D, L, scale, path, delta):
    """2×2 panel: (top-L) β vs cut, (top-R) E vs cut, with 95% CI bands;
                  (bot-L) residuals vs D by cut, (bot-R) LOO residuals vs D by cut."""
    cuts = sorted(results.keys())
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Cut-sweep diagnostics  (Huber δ = {delta:g})",
                 fontsize=FONT_TITLE + 2, y=1.00)
    ax_beta, ax_E, ax_resid, ax_loo = axes.flatten()

    # β and E vs cut
    for ax, key, idx, label in [(ax_beta, "β", 2, "β"), (ax_E, "E", 0, "E")]:
        ys = [results[k]["p"]["E" if key == "E" else "beta"] for k in cuts]
        mu = np.array([results[k]["boot_stats"][0][idx] for k in cuts])
        lo = np.array([results[k]["boot_stats"][2][idx] for k in cuts])
        hi = np.array([results[k]["boot_stats"][3][idx] for k in cuts])
        ax.plot(cuts, ys, "o-", color="tab:red", linewidth=2,
                markersize=10, zorder=5, label="point estimate")
        ax.fill_between(cuts, lo, hi, alpha=0.25, color="tab:blue",
                        label="bootstrap 95% CI")
        ax.plot(cuts, mu, "s--", color="tab:blue", alpha=0.7, label="bootstrap mean")
        ax.set_xlabel("k (num top-loss points dropped)", fontsize=FONT_LABEL)
        ax.set_ylabel(label, fontsize=FONT_LABEL)
        ax.set_xticks(cuts)
        ax.set_title(f"{label} vs outliers dropped", fontsize=FONT_TITLE)
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=FONT_LEGEND)

    # In-sample residuals vs D
    for i, k in enumerate(cuts):
        color = cmap(i / max(len(cuts) - 1, 1))
        r = results[k]
        ax_resid.scatter(r["Df"], r["in_sample_resid"], color=color, s=60,
                         edgecolors="k", linewidths=0.5, zorder=5,
                         label=f"k={k} (n={len(r['Df'])})")
        if len(r["Dd"]):
            ax_resid.scatter(r["Dd"], r["oob_dropped_resid"], color=color, s=80,
                             marker="X", edgecolors="k", linewidths=0.5,
                             zorder=6)
    ax_resid.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_resid.set_xscale("log", base=2)
    ax_resid.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_resid.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_resid.set_ylabel("log-loss residual  (log L − log pred)", fontsize=FONT_LABEL)
    ax_resid.set_title("Residuals by cut  (X = dropped/OOB point)", fontsize=FONT_TITLE)
    ax_resid.tick_params(labelsize=FONT_TICK)
    ax_resid.legend(fontsize=FONT_LEGEND)
    ax_resid.grid(alpha=0.3)

    # LOO residuals vs D
    for i, k in enumerate(cuts):
        color = cmap(i / max(len(cuts) - 1, 1))
        r = results[k]
        ax_loo.scatter(r["Df"], r["loo_resid"], color=color, s=60,
                       edgecolors="k", linewidths=0.5, zorder=5,
                       label=f"k={k}  LOO RMSE={rmse(r['loo_resid']):.3f}")
    ax_loo.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_loo.set_xscale("log", base=2)
    ax_loo.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_loo.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_loo.set_ylabel("LOO log-residual", fontsize=FONT_LABEL)
    ax_loo.set_title("Leave-one-out residuals", fontsize=FONT_TITLE)
    ax_loo.tick_params(labelsize=FONT_TICK)
    ax_loo.legend(fontsize=FONT_LEGEND)
    ax_loo.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(delta=DELTA, n_boot=N_BOOT):
    scale, D, L = extract_1epoch(ALL_DATASETS)
    loss_order = np.argsort(-L)  # descending by loss → top-k outliers

    print(f"\n{'='*92}")
    print("Dolma-30M 1-epoch  |  Chinchilla LSE fit (Besiroglu-style)  "
          f"|  δ={delta:g}  n_boot={n_boot}")
    print(f"{'='*92}")
    print("All 9 points, sorted by loss descending:")
    for i, idx in enumerate(loss_order):
        flag = "  ← top-" + str(i + 1) if i < 3 else ""
        print(f"  rank {i}: scale={scale[idx]:5.2f}x  D={D[idx]:.3e}  "
              f"L={L[idx]:.4f}{flag}")

    results = {}
    for k in [0, 1, 2, 3]:
        drop = loss_order[:k]
        keep = np.ones(len(D), dtype=bool); keep[drop] = False
        Df, Lf = D[keep], L[keep]
        Dd, Ld = D[~keep], L[~keep]

        print(f"\n--- k={k} dropped  |  n_fit={len(Df)}  "
              f"(scales: {sorted(scale[keep].tolist())}) ---")

        p = fit(Df, Lf, delta=delta)
        in_sample_resid = resid_log(Df, Lf, p)
        oob_dropped_resid = resid_log(Dd, Ld, p) if k else np.array([])

        init = {"e": p["e"], "b": p["b"], "beta": p["beta"]}
        loo = leave_one_out(Df, Lf, init, delta=delta)

        print(f"  Bootstrapping (n_boot={n_boot})...", flush=True)
        boot = bootstrap(Df, Lf, init, n_boot=n_boot, delta=delta)
        mu, se, lo, hi = summary_stats(boot)

        print(f"  n_valid_boot = {len(boot)}")
        print(f"  E  = {p['E']:8.3f}  |  boot  mean={mu[0]:.3f}  se={se[0]:.3f}  "
              f"95%CI=[{lo[0]:.3f}, {hi[0]:.3f}]")
        print(f"  B  = {p['B']:8.1f}  |  boot  mean={mu[1]:.1f}  se={se[1]:.1f}  "
              f"95%CI=[{lo[1]:.1f}, {hi[1]:.1f}]")
        print(f"  β  = {p['beta']:8.4f}  |  boot  mean={mu[2]:.4f}  se={se[2]:.4f}  "
              f"95%CI=[{lo[2]:.4f}, {hi[2]:.4f}]")
        print(f"  in-sample:  RMSE(logL)={rmse(in_sample_resid):.4f}  "
              f"max|Δlog|={maxabs(in_sample_resid):.4f}")
        print(f"  LOO:        RMSE(logL)={rmse(loo):.4f}  "
              f"max|Δlog|={maxabs(loo):.4f}")
        if k:
            print(f"  on dropped: RMSE(logL)={rmse(oob_dropped_resid):.4f}  "
                  f"max|Δlog|={maxabs(oob_dropped_resid):.4f}")

        results[k] = {
            "p": p, "Df": Df, "Lf": Lf, "Dd": Dd, "Ld": Ld,
            "in_sample_resid": in_sample_resid,
            "oob_dropped_resid": oob_dropped_resid,
            "loo_resid": loo, "boot": boot,
            "boot_stats": (mu, se, lo, hi),
        }

    tag = f"delta_{delta:g}".replace(".", "_").replace("-", "neg")
    plot_diagnostics(results, D, L, scale,
                     path=os.path.join(SCRIPT_DIR, f"tune_1epoch_30m_{tag}.pdf"),
                     delta=delta)

    # Concise final summary table
    print(f"\n{'='*92}")
    print("SUMMARY")
    print(f"{'='*92}")
    hdr = f"{'k':>2}  {'n':>2}  {'E':>6}  {'β':>7}  {'β_CI_width':>12}  " \
          f"{'RMSE_in':>8}  {'RMSE_LOO':>8}  {'RMSE_drop':>9}"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(results.keys()):
        r = results[k]
        mu, se, lo, hi = r["boot_stats"]
        ci_width = hi[2] - lo[2]
        rmse_drop = rmse(r["oob_dropped_resid"]) if k else float("nan")
        print(f"{k:>2}  {len(r['Df']):>2}  {r['p']['E']:6.3f}  {r['p']['beta']:.4f}  "
              f"{ci_width:12.4f}  {rmse(r['in_sample_resid']):8.4f}  "
              f"{rmse(r['loo_resid']):8.4f}  "
              f"{rmse_drop:9.4f}" if k else
              f"{k:>2}  {len(r['Df']):>2}  {r['p']['E']:6.3f}  {r['p']['beta']:.4f}  "
              f"{ci_width:12.4f}  {rmse(r['in_sample_resid']):8.4f}  "
              f"{rmse(r['loo_resid']):8.4f}  {'—':>9}")


if __name__ == "__main__":
    main(delta=1e-3)   # Besiroglu-style (effectively L1 on log-L)
    main(delta=0.1)    # L2-like on log-L
