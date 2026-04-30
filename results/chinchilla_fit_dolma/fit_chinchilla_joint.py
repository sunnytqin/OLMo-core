"""
Joint 5-parameter Chinchilla fit across all Dolma model sizes.

Functional form:

    L(N, D) = E + A / N^α + B / D^β

LSE parameterization:

    log L = logsumexp( e, a − α·log N, b − β·log D )

Same fit machinery as the single-size fits: L-BFGS + Huber(δ=0.1) on
log L, grid-search init, strong-Wolfe line search.

Uses all 1-epoch points from {14, 30, 60, 100, 190, 370, 600}M with the
same k=3 cut (scale ≥ 0.5×) as the per-size fits.
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))

from data import SIZES, DEFAULT_SCALE_MIN, TTP_RATIO, load, extract_1epoch  # noqa: E402
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.1
SCALE_MIN = DEFAULT_SCALE_MIN

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 16, 13, 11, 18

# Log-spaced Hoffmann-style grid for (e, a, b, α, β)
GRID = expand_grid({
    "e":     [-1.0, 0.0, 0.5, 1.0, 1.5],
    "a":     [0.0, 7.0, 14.0],
    "b":     [0.0, 7.0, 14.0],
    "alpha": [0.1, 0.3, 0.6],
    "beta":  [0.1, 0.3, 0.6],
})   # 5·3·3·3·3 = 405 inits


def per_size_1ep_fit(datasets, N, tag, scale_min=SCALE_MIN, delta=DELTA):
    """Drop the 5-param joint and refit per-size 3-param Chinchilla on this size's
    1-epoch data. Returns (E, B, β) or None if too few points."""
    from fit_lse import expand_grid as _eg
    grid = _eg({
        "e":    [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5],
        "b":    [0.0, 3.0, 7.0, 11.0, 15.0, 19.0],
        "beta": [0.05, 0.15, 0.3, 0.5, 0.7, 1.0],
    })
    scale, D, L = extract_1epoch(datasets, N, scale_min=scale_min)
    if len(D) < 3:
        return None
    log_D = torch.tensor(np.log(D), dtype=torch.float64)
    log_L = torch.tensor(np.log(L), dtype=torch.float64)

    def forward(p):
        terms = torch.stack([p["e"].expand_as(log_D),
                             p["b"] - p["beta"] * log_D], dim=0)
        return logsumexp_stable(terms, dim=0)

    res = fit_lse(forward, log_L, grid, delta=delta, verbose=False)
    p = res["params"]
    return dict(E=float(np.exp(p["e"])), B=float(np.exp(p["b"])),
                beta=float(p["beta"]), n=len(D), rmse=res["rmse_logL"])


def collect_1epoch_all_sizes(scale_min: float = SCALE_MIN):
    """Pool 1-epoch points across all sizes.

    Returns: (size_tag, N, scale, D, L) arrays, all length R.
    """
    tags, Ns, scales, Ds, Ls = [], [], [], [], []
    for size in SIZES:
        N, datasets = load(size)
        scale, D, L = extract_1epoch(datasets, N, scale_min=scale_min)
        for s, d, l in zip(scale, D, L):
            tags.append(size)
            Ns.append(N)
            scales.append(s)
            Ds.append(d)
            Ls.append(l)
    return (np.array(tags), np.array(Ns), np.array(scales),
            np.array(Ds), np.array(Ls))


def fit_joint(N_arr, D_arr, L_arr, delta: float = DELTA):
    log_N = torch.tensor(np.log(N_arr), dtype=torch.float64)
    log_D = torch.tensor(np.log(D_arr), dtype=torch.float64)
    log_L = torch.tensor(np.log(L_arr), dtype=torch.float64)

    def forward(p):
        terms = torch.stack([
            p["e"].expand_as(log_N),
            p["a"] - p["alpha"] * log_N,
            p["b"] - p["beta"] * log_D,
        ], dim=0)
        return logsumexp_stable(terms, dim=0)

    res = fit_lse(forward, log_L, GRID, delta=delta, verbose=False)
    p = res["params"]
    return dict(
        E=float(np.exp(p["e"])), A=float(np.exp(p["a"])),
        B=float(np.exp(p["b"])), alpha=float(p["alpha"]), beta=float(p["beta"]),
        _lse=res,
    )


def predict(N, D, p):
    return p["E"] + p["A"] / N ** p["alpha"] + p["B"] / D ** p["beta"]


def per_size_score(tags, N, D, L, p):
    rows = []
    for tag in sorted(set(tags)):
        m = tags == tag
        if not m.any():
            continue
        pred = predict(N[m], D[m], p)
        resid = np.log(L[m]) - np.log(pred)
        rows.append((tag, int(m.sum()), float(np.sqrt(np.mean(resid ** 2))),
                     float(np.max(np.abs(resid)))))
    return rows


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_joint(tags, N_arr, D_arr, L_arr, p, path):
    sizes = sorted(set(tags.tolist()), key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    ax_data, ax_res, ax_parity = axes

    # (1) L vs D coloured by size, with fit curves per size
    D_smooth = np.geomspace(D_arr.min() * 0.8, D_arr.max() * 1.2, 200)
    for i, tag in enumerate(sizes):
        m = tags == tag
        color = cmap(cnorm(i))
        ax_data.scatter(D_arr[m], L_arr[m], s=70, color=color,
                        edgecolors="k", linewidths=0.4, zorder=5,
                        label=f"{tag} (N={SIZES[tag][0]/1e6:.0f}M)")
        # Fit curve at this N
        L_curve = predict(SIZES[tag][0], D_smooth, p)
        ax_data.plot(D_smooth, L_curve, "-", color=color, linewidth=1.6, alpha=0.8)
    ax_data.set_xscale("log", base=2)
    ax_data.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_data.set_xlabel("Training tokens D", fontsize=FONT_LABEL)
    ax_data.set_ylabel("Validation loss", fontsize=FONT_LABEL)
    ax_data.set_title(f"Joint Chinchilla fit  "
                      f"(scale ≥ {SCALE_MIN}×, δ={DELTA})", fontsize=FONT_TITLE)
    ax_data.tick_params(labelsize=FONT_TICK)
    ax_data.legend(fontsize=FONT_LEGEND, loc="upper right", title="model")
    ax_data.grid(alpha=0.3)

    # (2) Residuals (log L obs − pred) vs D, by size
    pred = predict(N_arr, D_arr, p)
    resid = np.log(L_arr) - np.log(pred)
    for i, tag in enumerate(sizes):
        m = tags == tag
        color = cmap(cnorm(i))
        ax_res.scatter(D_arr[m], resid[m], s=70, color=color,
                       edgecolors="k", linewidths=0.4, zorder=5, label=tag)
    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_res.set_xlabel("Training tokens D", fontsize=FONT_LABEL)
    ax_res.set_ylabel("log L residual  (obs − pred)", fontsize=FONT_LABEL)
    ax_res.set_title("Residuals by size", fontsize=FONT_TITLE)
    ax_res.tick_params(labelsize=FONT_TICK)
    ax_res.legend(fontsize=FONT_LEGEND, loc="best")
    ax_res.grid(alpha=0.3)

    # (3) parity plot
    for i, tag in enumerate(sizes):
        m = tags == tag
        color = cmap(cnorm(i))
        ax_parity.scatter(pred[m], L_arr[m], s=70, color=color,
                          edgecolors="k", linewidths=0.4, zorder=5, label=tag)
    lo, hi = min(L_arr.min(), pred.min()) * 0.95, max(L_arr.max(), pred.max()) * 1.05
    ax_parity.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6, linewidth=1.2)
    ax_parity.set_xlabel("Predicted L", fontsize=FONT_LABEL)
    ax_parity.set_ylabel("Observed L", fontsize=FONT_LABEL)
    ax_parity.set_title("Parity (pred vs obs)", fontsize=FONT_TITLE)
    ax_parity.tick_params(labelsize=FONT_TICK)
    ax_parity.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────

DELTA_SWEEP = [0.1, 1e-3]   # canonical + Besiroglu reference
DROP_K_SWEEP = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25]


def fit_and_report(N_arr, D_arr, L_arr, tags, delta):
    p = fit_joint(N_arr, D_arr, L_arr, delta=delta)
    pred = predict(N_arr, D_arr, p)
    resid = np.log(L_arr) - np.log(pred)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = 1 - np.sum(resid ** 2) / np.sum((np.log(L_arr) - np.log(L_arr).mean()) ** 2)
    max_abs = float(np.max(np.abs(resid)))
    lse = p["_lse"]["params"]
    print(f"  δ={delta:<8.0e}  "
          f"E={p['E']:.3f}  A={p['A']:.1f}  B={p['B']:.0f}  "
          f"α={p['alpha']:.4f}  β={p['beta']:.4f}  "
          f"RMSE={rmse:.4f}  max|Δ|={max_abs:.4f}  R²={r2:.4f}")
    return p, rmse, r2


def topk_residual_drop_sweep(tags, N_arr, D_arr, L_arr, k_values=DROP_K_SWEEP,
                              delta=DELTA, iterative=True):
    """Besiroglu-style residual-based top-k dropping.

    iterative=False (one-shot): rank by residual from a single full-data
    fit, then drop top-k for each k.

    iterative=True (greedy / IRLS-style): for each k, repeat
        fit → compute residual → drop the worst → fit
    k times, recomputing residuals after each removal. This handles the
    case where many small-scale points coherently bias the fit (one-shot
    can't see past the cluster).
    """
    mode = "iterative greedy" if iterative else "one-shot"
    print(f"\n{'─'*92}")
    print(f"Top-k residual drop sweep — {mode}  (δ={delta}, "
          f"all {len(L_arr)} 1-ep points)")
    print(f"{'─'*92}")

    # Initial fit on everything
    p0 = fit_joint(N_arr, D_arr, L_arr, delta=delta)
    pred0 = predict(N_arr, D_arr, p0)
    resid0 = np.log(L_arr) - np.log(pred0)

    rows = {}
    cumulative_dropped = []  # only used in iterative mode
    print(f"  {'k':>3s}  {'n':>3s}  {'E':>7s}  {'A':>7s}  {'B':>9s}  "
          f"{'α':>6s}  {'β':>6s}  {'RMSE':>7s}  {'max|Δ|':>8s}  {'R²':>6s}  "
          f"{'last drop':<s}")

    if not iterative:
        # one-shot: rank once, drop top-k for each k
        sort_idx_oneshot = np.argsort(-np.abs(resid0))
    else:
        sort_idx_oneshot = None

    last_k = 0
    for k in k_values:
        if k >= len(L_arr) - 6:
            continue
        if iterative:
            # extend cumulative_dropped to length k by greedy fit/drop
            while len(cumulative_dropped) < k:
                keep = np.ones(len(L_arr), dtype=bool)
                keep[cumulative_dropped] = False
                p_tmp = fit_joint(N_arr[keep], D_arr[keep], L_arr[keep],
                                  delta=delta)
                pred_tmp = predict(N_arr[keep], D_arr[keep], p_tmp)
                resid_tmp = np.log(L_arr[keep]) - np.log(pred_tmp)
                worst_local = int(np.argmax(np.abs(resid_tmp)))
                # convert local index -> global index
                kept_global = np.where(keep)[0]
                cumulative_dropped.append(int(kept_global[worst_local]))
            keep = np.ones(len(L_arr), dtype=bool)
            keep[cumulative_dropped[:k]] = False
            drops_so_far = list(cumulative_dropped[:k])
        else:
            keep = np.ones(len(L_arr), dtype=bool)
            if k > 0:
                keep[sort_idx_oneshot[:k]] = False
            drops_so_far = list(sort_idx_oneshot[:k])

        # final fit at this k
        p = fit_joint(N_arr[keep], D_arr[keep], L_arr[keep], delta=delta)
        pred_in = predict(N_arr[keep], D_arr[keep], p)
        resid_in = np.log(L_arr[keep]) - np.log(pred_in)
        rmse = float(np.sqrt(np.mean(resid_in ** 2)))
        max_abs = float(np.max(np.abs(resid_in)))
        r2 = (1 - np.sum(resid_in ** 2) /
              np.sum((np.log(L_arr[keep]) - np.log(L_arr[keep]).mean()) ** 2))

        if k > 0:
            scales = D_arr / (TTP_RATIO * N_arr)
            recent = drops_so_far[max(0, k - 3):k]
            tag = ", ".join(f"{tags[i]}/{scales[i]:.2g}x" for i in recent)
            if k > 3: tag = "...,  " + tag
        else:
            tag = "—"
        print(f"  {k:>3d}  {keep.sum():>3d}  {p['E']:>7.3f}  {p['A']:>7.1f}  "
              f"{p['B']:>9.0f}  {p['alpha']:>6.3f}  {p['beta']:>6.3f}  "
              f"{rmse:>7.4f}  {max_abs:>8.4f}  {r2:>6.3f}  {tag}")
        rows[k] = dict(p=p, rmse=rmse, r2=r2, max_abs=max_abs,
                       dropped=drops_so_far, n_kept=int(keep.sum()))
    sort_idx = np.array(cumulative_dropped if iterative
                        else sort_idx_oneshot[:max(k_values)])
    return rows, sort_idx, resid0


def main():
    tags, N_arr, scale_arr, D_arr, L_arr = collect_1epoch_all_sizes(scale_min=0.0)
    n_by_size = {t: int(np.sum(tags == t)) for t in sorted(set(tags))}

    print(f"\n{'='*92}")
    print(f"Joint Chinchilla fit  (1-epoch, ALL scales — residual-based outlier drop)")
    print(f"{'='*92}")
    print(f"Total 1-epoch points: {len(L_arr)}  (grid: {len(GRID)} init points)")
    for tag, n in n_by_size.items():
        print(f"  {tag:<6s}  N={SIZES[tag][0]/1e6:5.0f}M   n={n}")

    # δ sweep on full data
    print(f"\n{'─'*92}")
    print("δ sweep on joint Chinchilla (all data, no drop):")
    print(f"{'─'*92}")
    by_delta = {}
    for delta in DELTA_SWEEP:
        p, _, _ = fit_and_report(N_arr, D_arr, L_arr, tags, delta)
        by_delta[delta] = p

    # Top-k residual drop sweep at canonical δ
    drop_sweep, sort_idx, resid_full = topk_residual_drop_sweep(
        tags, N_arr, D_arr, L_arr, k_values=DROP_K_SWEEP, delta=DELTA)

    # Pick canonical k as smallest k where β stabilizes between adjacent
    # k values (|Δβ| < 0.01).  Rationale: when iterative drop has removed
    # the systematic small-scale bias, β stops moving even as more points
    # are dropped — that's the convergence point.
    k_sorted = sorted(drop_sweep.keys())
    betas = [drop_sweep[k]["p"]["beta"] for k in k_sorted]
    canonical_k = k_sorted[-1]   # fallback: most-aggressive drop
    for i in range(1, len(betas)):
        if abs(betas[i] - betas[i - 1]) < 0.01:
            canonical_k = k_sorted[i]
            break
    print(f"\nCanonical k (heuristic: smallest k with |Δβ|<0.01): k = {canonical_k}")
    p = drop_sweep[canonical_k]["p"]
    print(f"  E={p['E']:.4f}  A={p['A']:.2f}  B={p['B']:.2f}  "
          f"α={p['alpha']:.4f}  β={p['beta']:.4f}  "
          f"RMSE={drop_sweep[canonical_k]['rmse']:.4f}")

    # For comparison: also report the legacy scale-based cut
    print(f"\nFor comparison: legacy fit (scale ≥ 0.5x):")
    keep_legacy = scale_arr >= 0.5
    p_legacy = fit_joint(N_arr[keep_legacy], D_arr[keep_legacy],
                         L_arr[keep_legacy], delta=DELTA)
    pred_legacy = predict(N_arr[keep_legacy], D_arr[keep_legacy], p_legacy)
    rmse_legacy = float(np.sqrt(np.mean(
        (np.log(L_arr[keep_legacy]) - np.log(pred_legacy)) ** 2)))
    print(f"  n={keep_legacy.sum()}  E={p_legacy['E']:.4f}  "
          f"A={p_legacy['A']:.2f}  B={p_legacy['B']:.2f}  "
          f"α={p_legacy['alpha']:.4f}  β={p_legacy['beta']:.4f}  "
          f"RMSE={rmse_legacy:.4f}")

    # Per-size residuals + implied E_eff at canonical-k fit
    print(f"\nPer-size residuals at canonical k={canonical_k}:")
    print(f"  {'size':<6s}  {'n':>3s}  {'RMSE':>8s}  {'max|Δ|':>8s}  "
          f"{'E_eff(N)':>9s}")
    keep = np.ones(len(L_arr), dtype=bool)
    if canonical_k > 0:
        keep[sort_idx[:canonical_k]] = False
    for tag, n, rm, mx in per_size_score(
            tags[keep], N_arr[keep], D_arr[keep], L_arr[keep], p):
        Nt = SIZES[tag][0]
        E_eff = p["E"] + p["A"] / Nt ** p["alpha"]
        print(f"  {tag:<6s}  {n:>3d}  {rm:>8.4f}  {mx:>8.4f}  "
              f"{E_eff:>9.4f}")

    # Per-size standalone 3-param fits (for comparison with joint-implied E_eff)
    print(f"\nPer-size 3-param 1-epoch fits  (each size fit independently, all scales):")
    print(f"  {'size':<6s}  {'n':>3s}  {'E':>8s}  {'B':>11s}  "
          f"{'β':>7s}  {'RMSE':>8s}")
    for tag in sorted(set(tags), key=lambda t: SIZES[t][0]):
        Nt, ds = load(tag)
        fit = per_size_1ep_fit(ds, Nt, tag, scale_min=0.0, delta=DELTA)
        if fit is None:
            print(f"  {tag:<6s}  (<3 pts — skipped)")
            continue
        print(f"  {tag:<6s}  {fit['n']:>3d}  {fit['E']:>8.4f}  "
              f"{fit['B']:>11.2f}  {fit['beta']:>7.4f}  "
              f"{fit['rmse']:>8.4f}")

    plot_joint(tags[keep], N_arr[keep], D_arr[keep], L_arr[keep], p,
               path=os.path.join(SCRIPT_DIR, "fit_chinchilla_joint.pdf"))

    return p, drop_sweep, sort_idx


if __name__ == "__main__":
    main()
