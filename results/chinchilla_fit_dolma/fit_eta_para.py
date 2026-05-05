"""
η_para fitting for paraphrase Dolma runs, across model sizes 14M / 30M / 60M / 190M.

Same framework as the multi-epoch η fit (fit_eta.py), but the second
token stream is *paraphrased* tokens rather than repeats:

    L = E_eff(N) + B / (D + η_para(D, D'; N) · D')^β

where
    D       = scale · 20 · N        (fresh Dolma tokens)
    D'_para = tokens_trained − D    (paraphrased tokens added on top)

We fix (E, A, B, α, β) at the joint Chinchilla anchors from
fit_chinchilla_joint.py and fit only η_para — exactly mirroring the
two-stage repetition pipeline.

Functional form is decided: Form B (exp-sat / Muennighoff Eq 5)
  η · D'/D = R*(1 − e^{−x/R*}),   x = D'/D,
  log R*(D, N) = log_K + ρ · log(D/N) + σ · log N         [3-param]

We also report Hill, tanh and constant-R* variants as cross-checks,
plus the legacy 2-param "Muennighoff Eq 5" (no σ) for direct
comparison with the repetition writeup.
"""

import glob
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))

# ── Palatino (URW P052 clone) — match paper_figures.py styling ──────
for _f in glob.glob("/usr/share/fonts/urw-base35/P052-*.otf"):
    font_manager.fontManager.addfont(_f)

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["P052", "Palatino", "TeX Gyre Pagella", "serif"],
    "mathtext.fontset": "cm",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.titlesize": 14,
})

from data import (DEFAULT_SCALE_MIN, SIZES, TTP_RATIO,  # noqa: E402
                  extract_paraphrase, load_with_para)
from fit_chinchilla_joint import (collect_1epoch_all_sizes, fit_joint,  # noqa: E402
                                   topk_residual_drop_sweep)
from fit_eta import (FORMS, fit_form, leave_one_out, per_point_eta,  # noqa: E402
                     per_point_eta_diff)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 1e-3
SCALE_MIN = 0.5  # match fit_eta.py default for the η-fit step

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 15, 12, 10, 17

PARA_SIZES = ["14m", "30m", "60m", "190m"]


# ──────────────────────────────────────────────────────────────────────
# Data pooling
# ──────────────────────────────────────────────────────────────────────

def collect_paraphrase_all_sizes(scale_min: float = SCALE_MIN):
    """Pool paraphrase points across all sizes that have parap_datasets."""
    tags, Ns, scales, Ds, Ks, Dps, Ls, L_1eps = [], [], [], [], [], [], [], []
    for size in PARA_SIZES:
        N, datasets, parap = load_with_para(size)
        if not parap:
            continue
        s, D, K, Dp, L, L_1ep = extract_paraphrase(
            datasets, parap, N, scale_min=scale_min)
        if len(D) == 0:
            continue
        tags.extend([size] * len(D))
        Ns.extend([N] * len(D))
        scales.extend(s)
        Ds.extend(D)
        Ks.extend(K)
        Dps.extend(Dp)
        Ls.extend(L)
        L_1eps.extend(L_1ep)
    return (np.array(tags), np.array(Ns, dtype=np.float64),
            np.array(scales), np.array(Ds), np.array(Ks),
            np.array(Dps), np.array(Ls), np.array(L_1eps))


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_per_size_eta_para(results, path, title,
                           forms=("Muennighoff R*(N)", "Muennighoff Eq 5",
                                  "tanh R*(N)", "Hill R*(N)")):
    sizes = sorted(results.keys(), key=lambda t: SIZES[t][0])
    fig, axes = plt.subplots(len(forms), len(sizes),
                             figsize=(3.4 * len(sizes), 2.7 * len(forms)),
                             squeeze=False)
    for row, form_name in enumerate(forms):
        for col, size in enumerate(sizes):
            ax = axes[row, col]
            d = results[size]
            r = d.get(form_name)
            if r is None:
                ax.set_visible(False)
                continue
            ax.scatter(d["Dp"] / d["D"], r["eta_pred"], s=22, color="tab:blue",
                       edgecolors="k", linewidths=0.3, zorder=5,
                       label="parametric η")
            v = ~np.isnan(d["eta_pp"])
            ax.scatter(d["Dp"][v] / d["D"][v], d["eta_pp"][v], s=22,
                       color="gray", edgecolors="k", linewidths=0.2,
                       alpha=0.55, zorder=4, label="per-point η (ΔL)")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"{size}  (N={SIZES[size][0]/1e6:.0f}M)",
                             fontsize=FONT_LEGEND)
            if col == 0:
                ax.set_ylabel(f"{form_name}\nη_para", fontsize=FONT_LEGEND)
            if row == len(forms) - 1:
                ax.set_xlabel("D'/D", fontsize=FONT_LEGEND)
            ax.tick_params(labelsize=FONT_TICK - 2)
            ax.grid(alpha=0.25, which="both")
            loo = r.get("loo_rmse", float("nan"))
            ax.text(0.03, 0.03, f"LOO={loo:.3f}",
                    transform=ax.transAxes, fontsize=FONT_LEGEND - 2,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", alpha=0.8))
    axes[0, 0].legend(fontsize=FONT_LEGEND - 2, loc="upper right")
    fig.suptitle(title, fontsize=FONT_TITLE + 1, y=1.00)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_joint_eta_para(tags, D, Dp, N_arr, eta_pp, res, form_name, path):
    sizes = sorted(set(tags), key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_eta, ax_res = axes

    for i, size in enumerate(sizes):
        m = tags == size
        color = cmap(cnorm(i))
        v = m & ~np.isnan(eta_pp)
        ax_eta.scatter(Dp[v] / D[v], eta_pp[v], s=50, color=color,
                       edgecolors="k", linewidths=0.3, alpha=0.6,
                       label=f"{size}  (per-point)")
        ax_eta.scatter(Dp[m] / D[m], res["eta_pred"][m], s=40, color=color,
                       marker="x", linewidths=1.3)
        ax_res.scatter(Dp[m] / D[m], res["resid"][m], s=50, color=color,
                       edgecolors="k", linewidths=0.3, label=f"{size}")

    ax_eta.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_eta.set_xscale("log")
    ax_eta.set_yscale("log")
    ax_eta.set_xlabel("D'_para / D", fontsize=FONT_LABEL)
    ax_eta.set_ylabel("η_para", fontsize=FONT_LABEL)
    ax_eta.set_title(f"Joint η_para fit — {form_name}\n"
                     f"(circles = per-point ΔL; × = fitted)",
                     fontsize=FONT_TITLE)
    ax_eta.legend(fontsize=FONT_LEGEND - 1, loc="best", ncol=2)
    ax_eta.grid(alpha=0.3, which="both")

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("D'_para / D", fontsize=FONT_LABEL)
    ax_res.set_ylabel("log L residual  (obs − pred)", fontsize=FONT_LABEL)
    ax_res.set_title(f"Residuals  (RMSE={res['rmse']:.4f}, "
                     f"LOO={res.get('loo_rmse', float('nan')):.4f})",
                     fontsize=FONT_TITLE)
    ax_res.legend(fontsize=FONT_LEGEND - 1, loc="best")
    ax_res.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_para_vs_repeat(tags, D_p, Dp_p, eta_pp_para,
                        tags_r, D_r, Dp_r, eta_pp_repeat,
                        path, anchor):
    """Side-by-side per-point η: paraphrase (left) vs repetition (right),
    on the same axes for easy comparison."""
    sizes = sorted(set(tags) | set(tags_r), key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax_p, ax_r = axes

    for i, size in enumerate(sizes):
        color = cmap(cnorm(i))
        m_p = tags == size
        v_p = m_p & ~np.isnan(eta_pp_para)
        if v_p.any():
            ax_p.scatter(Dp_p[v_p] / D_p[v_p], eta_pp_para[v_p], s=55,
                         color=color, edgecolors="k", linewidths=0.3,
                         alpha=0.75, label=size)
        m_r = tags_r == size
        v_r = m_r & ~np.isnan(eta_pp_repeat)
        if v_r.any():
            ax_r.scatter(Dp_r[v_r] / D_r[v_r], eta_pp_repeat[v_r], s=55,
                         color=color, edgecolors="k", linewidths=0.3,
                         alpha=0.75, label=size)

    for ax, label in [(ax_p, "Paraphrase  (η_para = effective fresh-equiv "
                              "of paraphrased tokens)"),
                       (ax_r, "Repetition  (η_repeat = fresh-equiv of "
                              "repeated tokens)")]:
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("D'/D", fontsize=FONT_LABEL)
        ax.set_title(label, fontsize=FONT_LEGEND + 1)
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=FONT_LEGEND, loc="best", ncol=2)
    ax_p.set_ylabel("η  (per-point ΔL solver)", fontsize=FONT_LABEL)

    fig.suptitle(f"Per-point η — paraphrase vs repetition  "
                 f"(B={anchor['B']:.0f}, β={anchor['beta']:.3f})",
                 fontsize=FONT_TITLE, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

_ANCHOR_CACHE = os.path.join(SCRIPT_DIR, ".joint_anchor_cache.npz")


def get_joint_anchors(verbose: bool = True, canonical_k: int = 20,
                       use_cache: bool = True):
    """Refit joint Chinchilla and return the anchors at a fixed canonical k.

    Default k=20 matches the canonical anchor in the repetition writeup
    (E=1.72, A=1115, B=20828, α=0.390, β=0.451) — reuses the same Chinchilla
    curve so η_para and η_repeat are directly comparable.

    Result is cached to .joint_anchor_cache.npz so the slow iterative
    residual-drop sweep only runs once across iterations.
    """
    if use_cache and os.path.exists(_ANCHOR_CACHE):
        z = np.load(_ANCHOR_CACHE)
        anchor = {k: float(z[k]) for k in ["E", "A", "B", "alpha", "beta"]}
        if verbose:
            print(f"\nLoaded cached joint Chinchilla anchors "
                  f"(canonical k={int(z['k'])}):")
            print(f"  E={anchor['E']:.4f}  A={anchor['A']:.2f}  "
                  f"B={anchor['B']:.2f}  α={anchor['alpha']:.4f}  "
                  f"β={anchor['beta']:.4f}")
        return anchor

    tags_1ep, N_1ep, _, D_1ep, L_1ep = collect_1epoch_all_sizes(scale_min=0.0)
    # Use the same fine k_values list as the repetition writeup (§1.1), so
    # the |Δβ|<0.01 break selects the same canonical_k=20 anchor here.
    drop_sweep, _, _ = topk_residual_drop_sweep(
        tags_1ep, N_1ep, D_1ep, L_1ep,
        k_values=[0, 1, 2, 3, 5, 8, 12, 16, 20, 25], delta=0.1, iterative=True)
    if canonical_k not in drop_sweep:
        k_sorted = sorted(drop_sweep.keys())
        betas_seq = [drop_sweep[k]["p"]["beta"] for k in k_sorted]
        canonical_k = k_sorted[-1]
        for i in range(1, len(betas_seq)):
            if abs(betas_seq[i] - betas_seq[i - 1]) < 0.01:
                canonical_k = k_sorted[i]
                break
    anchor = drop_sweep[canonical_k]["p"]
    if verbose:
        print(f"\nUsing canonical k={canonical_k} anchors:")
        print(f"  E={anchor['E']:.4f}  A={anchor['A']:.2f}  "
              f"B={anchor['B']:.2f}  α={anchor['alpha']:.4f}  "
              f"β={anchor['beta']:.4f}")
    np.savez(_ANCHOR_CACHE,
             E=anchor["E"], A=anchor["A"], B=anchor["B"],
             alpha=anchor["alpha"], beta=anchor["beta"], k=canonical_k)
    return anchor


def main():
    print(f"\n{'='*100}")
    print("Joint Chinchilla anchors (reused from fit_eta.py)")
    print(f"{'='*100}")
    anchor = get_joint_anchors()
    E_j, A_j, B_j = anchor["E"], anchor["A"], anchor["B"]
    alpha_j, beta_j = anchor["alpha"], anchor["beta"]

    def E_eff(N):
        return E_j + A_j / N ** alpha_j

    # ─── 1) Per-size η_para fits ─────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"Per-size η_para fits  (paraphrase data, scale ≥ {SCALE_MIN}×, "
          f"δ={DELTA})")
    print(f"{'='*100}")

    per_size: Dict[str, Dict] = {}
    for size in PARA_SIZES:
        N, datasets, parap = load_with_para(size)
        if not parap:
            print(f"  {size}: no parap_datasets — skipping")
            continue
        s, D, K, Dp, L, L_1ep = extract_paraphrase(
            datasets, parap, N, scale_min=SCALE_MIN)
        if len(D) < 5:
            print(f"  {size}: only {len(D)} paraphrase points — skipping")
            continue
        E_size = E_eff(N)
        eta_pp = per_point_eta_diff(D, Dp, L, L_1ep, B_j, beta_j)
        eta_pp_legacy = per_point_eta(D, Dp, L, E_size, B_j, beta_j)
        n_gt1 = int(np.sum(eta_pp > 1))
        n_lt0 = int(np.sum(eta_pp < 0))
        valid = ~np.isnan(eta_pp)
        print(f"\n  {size}  (N={N/1e6:.0f}M, n={len(D)}, "
              f"E_eff={E_size:.3f}):")
        print(f"    per-point η_para (ΔL form):  "
              f"min={np.nanmin(eta_pp):.3f} "
              f"median={np.nanmedian(eta_pp):.3f} "
              f"max={np.nanmax(eta_pp):.3f}  "
              f"(η>1: {n_gt1}/{int(valid.sum())}, "
              f"η<0: {n_lt0}/{int(valid.sum())})")
        print(f"    per-point η_para (E_eff form): "
              f"min={np.nanmin(eta_pp_legacy):.3f} "
              f"median={np.nanmedian(eta_pp_legacy):.3f} "
              f"max={np.nanmax(eta_pp_legacy):.3f}  (legacy)")
        print(f"    {'form':<22s}  {'RMSE':>7s}  {'LOO':>7s}  "
              f"{'R²':>6s}  params")
        per_size[size] = {"D": D, "Dp": Dp, "K": K, "scale": s,
                          "eta_pp": eta_pp, "eta_pp_legacy": eta_pp_legacy,
                          "L": L, "L_1ep": L_1ep, "N": N}
        for name, form in FORMS.items():
            try:
                res = fit_form(form, D, Dp, N, L, E_size, B_j, beta_j,
                               delta=DELTA)
                loo = leave_one_out(form, D, Dp, N, L, E_size, B_j, beta_j,
                                    delta=DELTA, warm_init=res["params"])
                v_loo = ~np.isnan(loo)
                res["loo_rmse"] = (float(np.sqrt(np.mean(loo[v_loo] ** 2)))
                                    if v_loo.any() else float("nan"))
            except RuntimeError as e:
                print(f"    {name:<22s}  failed: {e}")
                continue
            per_size[size][name] = res
            pstr = "  ".join(f"{k}={v:.3g}" for k, v in res["params"].items())
            print(f"    {name:<22s}  {res['rmse']:>7.4f}  "
                  f"{res['loo_rmse']:>7.4f}  {res['r2']:>6.3f}  {pstr}")

    # ─── 2) Joint η_para fit (pooled) ────────────────────────────────
    print(f"\n{'='*100}")
    print("Joint η_para fit across all sizes  (pooled paraphrase data)")
    print(f"{'='*100}")
    tags, Ns, scales, Ds, Ks, Dps, Ls, L_1eps = collect_paraphrase_all_sizes(
        SCALE_MIN)
    E_eff_arr = np.array([E_eff(n) for n in Ns])
    eta_pp_pool = per_point_eta_diff(Ds, Dps, Ls, L_1eps, B_j, beta_j)
    eta_pp_pool_legacy = per_point_eta(Ds, Dps, Ls, E_eff_arr, B_j, beta_j)
    valid_pool = ~np.isnan(eta_pp_pool)
    n_gt1 = int(np.sum(eta_pp_pool[valid_pool] > 1))
    n_lt0 = int(np.sum(eta_pp_pool[valid_pool] < 0))
    print(f"  pooled n = {len(Ds)} across sizes: "
          f"{sorted(set(tags.tolist()))}")
    print(f"  per-point η_para (ΔL):    "
          f"min={np.nanmin(eta_pp_pool):.3f} "
          f"median={np.nanmedian(eta_pp_pool):.3f} "
          f"max={np.nanmax(eta_pp_pool):.3f} "
          f"(η>1: {n_gt1}/{int(valid_pool.sum())}, "
          f"η<0: {n_lt0}/{int(valid_pool.sum())})")
    print(f"  per-point η_para (E_eff): "
          f"min={np.nanmin(eta_pp_pool_legacy):.3f} "
          f"median={np.nanmedian(eta_pp_pool_legacy):.3f} "
          f"max={np.nanmax(eta_pp_pool_legacy):.3f}")
    print(f"\n  {'form':<22s}  {'RMSE':>7s}  {'LOO':>7s}  "
          f"{'R²':>6s}  params")

    joint_results: Dict[str, Dict] = {}
    for name, form in FORMS.items():
        try:
            res = fit_form(form, Ds, Dps, Ns, Ls, E_eff_arr, B_j, beta_j,
                           delta=DELTA)
            loo = leave_one_out(form, Ds, Dps, Ns, Ls, E_eff_arr, B_j, beta_j,
                                delta=DELTA, warm_init=res["params"])
            res["loo_rmse"] = float(np.sqrt(np.mean(loo ** 2)))
        except RuntimeError as e:
            print(f"  {name:<22s}  failed: {e}")
            continue
        joint_results[name] = res
        pstr = "  ".join(f"{k}={v:.3g}" for k, v in res["params"].items())
        print(f"  {name:<22s}  {res['rmse']:>7.4f}  "
              f"{res['loo_rmse']:>7.4f}  {res['r2']:>6.3f}  {pstr}")

    best_joint = min(joint_results, key=lambda k: joint_results[k]["loo_rmse"])
    print(f"\nBest joint form: {best_joint}   "
          f"({FORMS[best_joint]['desc']})")

    # ─── 3) Plots ────────────────────────────────────────────────────
    plot_per_size_eta_para(
        per_size, path=os.path.join(SCRIPT_DIR, "fit_eta_para_per_size.pdf"),
        title=f"η_para per-size fits  "
              f"(δ={DELTA}, scale ≥ {SCALE_MIN}×)")

    headline = "Muennighoff R*(N)"
    if headline not in joint_results:
        headline = best_joint
    plot_joint_eta_para(
        tags, Ds, Dps, Ns, eta_pp_pool, joint_results[headline], headline,
        path=os.path.join(SCRIPT_DIR, "fit_eta_para_joint.pdf"))

    # Comparison to repetition: use the existing fit_eta machinery's data
    # collector so we get matched per-point η_repeat under the same anchors.
    from fit_eta import collect_multi_epoch_all_sizes
    tags_r, Ns_r, _, Ds_r, eps_r, Dps_r, Ls_r, L_1eps_r = (
        collect_multi_epoch_all_sizes(SCALE_MIN))
    eta_pp_repeat = per_point_eta_diff(Ds_r, Dps_r, Ls_r, L_1eps_r, B_j, beta_j)
    plot_para_vs_repeat(
        tags, Ds, Dps, eta_pp_pool,
        tags_r, Ds_r, Dps_r, eta_pp_repeat,
        path=os.path.join(SCRIPT_DIR, "fit_eta_para_vs_repeat.pdf"),
        anchor=anchor)

    return per_size, joint_results, anchor


if __name__ == "__main__":
    main()
