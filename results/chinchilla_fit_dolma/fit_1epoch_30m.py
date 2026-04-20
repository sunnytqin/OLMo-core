"""
Classic 1-epoch Chinchilla fit on Dolma 30M data.

Functional form (one model size, so the N term collapses into E):

    L(D) = E + B / D^beta              [additive]
    log L = LSE(e, b - beta * log D)   [LSE, algebraically identical]

Fits with L-BFGS + Huber on log L, grid-search init, across a δ sweep.
Smaller δ → more L1-like; larger δ → more L2-like.

Also produces a log-log diagnostic: (L − Ê) vs D.  On a clean power law
this is a straight line with slope −β; deviations highlight outliers.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

# Path to results/ for dolma_30m + shared fitter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dolma_30m import ALL_DATASETS  # noqa: E402
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402

N = 30e6
TTP_RATIO = 20
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 18, 16, 13, 20


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def extract_1epoch(datasets):
    """Return (scale, D, loss) arrays for all 1-epoch points."""
    rows = []
    for ds in datasets:
        scale = ds["chinchilla_scale"][0]
        idx = ds["epochs"].index(1)
        loss = ds["validation_loss"][idx]
        if not np.isnan(loss):
            rows.append((scale, scale * TTP_RATIO * N, loss))
    rows = np.array(rows)
    return rows[:, 0], rows[:, 1], rows[:, 2]


# ──────────────────────────────────────────────────────────────────────
# Fit
# ──────────────────────────────────────────────────────────────────────

def classic_law(D, E, B, beta):
    return E + B / D ** beta


GRID = expand_grid({
    "e":    [-1.0, 0.0, 0.5, 1.0, 1.5],
    "b":    [2.0, 5.0, 8.0, 11.0, 14.0],
    "beta": [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5],
})


def fit_lse_raw(D, L, delta):
    log_D = torch.tensor(np.log(D), dtype=torch.float64)
    log_L = torch.tensor(np.log(L), dtype=torch.float64)

    def forward(p):
        terms = torch.stack([p["e"].expand_as(log_D),
                             p["b"] - p["beta"] * log_D], dim=0)
        return logsumexp_stable(terms, dim=0)

    res = fit_lse(forward, log_L, GRID, delta=delta, verbose=False)
    p = res["params"]
    return dict(E=float(np.exp(p["e"])), B=float(np.exp(p["b"])),
                beta=float(p["beta"]))


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────

def score(params, D, L):
    pred = classic_law(D, params["E"], params["B"], params["beta"])
    log_L, log_pred = np.log(L), np.log(pred)
    return dict(
        rmse=np.sqrt(np.mean((L - pred) ** 2)),
        r2=1 - np.sum((L - pred) ** 2) / np.sum((L - L.mean()) ** 2),
        rmse_log=np.sqrt(np.mean((log_L - log_pred) ** 2)),
        r2_log=1 - (np.sum((log_L - log_pred) ** 2)
                    / np.sum((log_L - log_L.mean()) ** 2)),
        max_abs_log_resid=np.max(np.abs(log_L - log_pred)),
    )


def print_row(name, p, s):
    print(f"  {name:<22s}  E={p['E']:6.3f}  B={p['B']:>11.2f}  β={p['beta']:.4f}  "
          f"RMSE(L)={s['rmse']:.5f}  RMSE(logL)={s['rmse_log']:.5f}  "
          f"R²(L)={s['r2']:.4f}  R²(logL)={s['r2_log']:.4f}  "
          f"max|Δlog|={s['max_abs_log_resid']:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_fits(D, L, scale, fits, ref_E, path, title, min_scale):
    """Two-panel figure:
        Left:  L vs D (log-x, linear-y) with all fit curves.
        Right: log-log of (L − ref_E) vs D -- a clean power law is a line.
    """
    excluded = scale < min_scale
    D_smooth = np.geomspace(D.min() * 0.5, D.max() * 2, 300)
    fig, (ax, ax_ll) = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left: linear loss ──
    ax.scatter(D[excluded], L[excluded], s=80, color="lightgray",
               edgecolors="k", linewidths=0.5, zorder=5,
               label=f"< {min_scale}x (excluded)")
    ax.scatter(D[~excluded], L[~excluded], s=90, color="tab:blue",
               edgecolors="k", linewidths=0.5, zorder=6, label="fit data")
    for scale_i, D_i, L_i in zip(scale, D, L):
        ax.annotate(f"{scale_i:g}x", (D_i, L_i),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=FONT_LEGEND - 3, color="gray")

    cmap = plt.cm.viridis
    for i, (name, p) in enumerate(fits.items()):
        color = cmap(i / max(len(fits) - 1, 1))
        ax.plot(D_smooth, classic_law(D_smooth, p["E"], p["B"], p["beta"]),
                "-", color=color, linewidth=1.8,
                label=f"{name}: E={p['E']:.2f}, β={p['beta']:.3f}")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND - 1, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Right: log-log residual form ──
    resid = L - ref_E
    valid = resid > 0
    incl = valid & ~excluded
    excl = valid & excluded
    ax_ll.scatter(D[excl], resid[excl], s=80, color="lightgray",
                  edgecolors="k", linewidths=0.5, zorder=5,
                  label=f"< {min_scale}x (excluded)")
    ax_ll.scatter(D[incl], resid[incl], s=90, color="tab:blue",
                  edgecolors="k", linewidths=0.5, zorder=6,
                  label=f"L − Ê   (Ê = {ref_E:.3f})")
    for scale_i, D_i, r_i in zip(scale[valid], D[valid], resid[valid]):
        ax_ll.annotate(f"{scale_i:g}x", (D_i, r_i),
                       textcoords="offset points", xytext=(6, 6),
                       fontsize=FONT_LEGEND - 3, color="gray")
    for i, (name, p) in enumerate(fits.items()):
        color = cmap(i / max(len(fits) - 1, 1))
        pred_resid = p["B"] / D_smooth ** p["beta"]
        ax_ll.plot(D_smooth, pred_resid, "-", color=color, linewidth=1.8,
                   label=f"{name}: slope −{p['beta']:.3f}")
    ax_ll.set_xscale("log", base=2)
    ax_ll.set_yscale("log")
    ax_ll.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_ll.set_xlabel("Fresh tokens D", fontsize=FONT_LABEL)
    ax_ll.set_ylabel("L − Ê   (log scale)", fontsize=FONT_LABEL)
    ax_ll.set_title("Power-law diagnostic (log-log)", fontsize=FONT_TITLE)
    ax_ll.tick_params(labelsize=FONT_TICK)
    ax_ll.legend(fontsize=FONT_LEGEND - 1, loc="upper right")
    ax_ll.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

DELTAS = [1.0, 0.1, 0.03, 0.01, 1e-3, 1e-5]


def run(min_scale):
    print(f"\n{'='*90}")
    print(f"Dolma-30M 1-epoch fit  (scales >= {min_scale}x)")
    print(f"{'='*90}")

    scale, D, L = extract_1epoch(ALL_DATASETS)
    mask = scale >= min_scale
    Df, Lf = D[mask], L[mask]
    print(f"  Fitting on {mask.sum()}/{len(mask)} points  "
          f"(scales: {sorted(scale[mask].tolist())})")

    fits = {}
    for delta in DELTAS:
        fits[f"LSE δ={delta:g}"] = fit_lse_raw(Df, Lf, delta=delta)

    print(f"\n  {'method':<22s}  params + errors (evaluated on fit points)")
    for name, p in fits.items():
        print_row(name, p, score(p, Df, Lf))

    # Reference E for the log-log diagnostic: use the cleanest (smallest-δ) LSE fit
    ref_E = fits[f"LSE δ={DELTAS[-1]:g}"]["E"]

    tag = f"min{str(min_scale).replace('.', '_')}x"
    plot_fits(D, L, scale, fits, ref_E,
              path=os.path.join(SCRIPT_DIR, f"fit_1epoch_30m_{tag}.pdf"),
              title=f"Dolma-30M  1-epoch fits  (min {min_scale}x)",
              min_scale=min_scale)
    return fits


if __name__ == "__main__":
    # min_scale cuts map to top-k dropped since the smallest-scale points are
    # the highest-loss ones in this dataset:
    #   min_scale=0.0  -> k=0 (keep all 9)
    #   min_scale=0.1  -> k=1 (drop 0.05x)
    #   min_scale=0.25 -> k=2 (drop 0.05x, 0.1x)
    #   min_scale=0.5  -> k=3 (drop 0.05x, 0.1x, 0.25x)
    run(min_scale=0.0)
    run(min_scale=0.1)
    run(min_scale=0.25)
    run(min_scale=0.5)
