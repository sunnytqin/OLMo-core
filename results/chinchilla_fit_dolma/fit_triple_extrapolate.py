"""
Extrapolation test for §6's triple-joint fit.

We refit the 11-parameter triple model on **only N ≤ 30M runs**
(14M + 30M, 1-epoch + repetition + paraphrase) and use the resulting
parameters to *predict* validation losses on the held-out sizes
{190M, 370M, 600M} across all D' regimes.

If the law generalises out of sample, the extrapolated predictions
should track the observed losses with RMSE comparable to the
in-sample fit's RMSE.

Output:
  fit_triple_extrapolate.pdf — parity + per-N residual diagnostic.
  Tabular comparison: in-sample (small-N) vs held-out (large-N) RMSE.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))

# Palatino styling
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

from data import SIZES, TTP_RATIO  # noqa: E402
from fit_joint_triple import (DELTA, SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA,  # noqa: E402
                                _print_params, collect_pooled_triple,
                                make_triple_forward, stage1_rep_only,
                                stage2_triple, topk_drop_sweep)
from fit_lse import fit_lse  # noqa: E402

# §6 TRIPLE k=15 anchors (writeup §6.2): used as warm-start so the
# small-N (only 2-point N axis) fit doesn't wander into a degenerate
# (E, A, α) basin where α drifts to physically-implausible values.
TRIPLE_ANCHOR = dict(
    E=0.003, A=28.9, B=15599, alpha=0.133, beta=0.431,
    log_K_rep=10.58, rho_rep=-0.414, sigma_rep=-0.394,
    log_K_para=10.10, rho_para=-2.555, sigma_para=+0.177,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SMALL_N_CUT = 30e6 + 1.0   # include 14M + 30M, exclude 60M
HELDOUT_TAGS = ("190m", "370m", "600m")
ALL_HELDOUT_TAGS = ("60m", "100m", "190m", "370m", "600m")  # for context


def filter_data(data, mask):
    return {
        "tags":   data["tags"][mask],
        "N":      data["N"][mask],
        "D":      data["D"][mask],
        "Dp":     data["Dp"][mask],
        "L":      data["L"][mask],
        "source": data["source"][mask],
    }


def predict_log_L(params, N, D, Dp, source):
    """Run the triple forward at given params on the given arrays."""
    fwd = make_triple_forward(N, D, Dp, source)
    p_t = {k: torch.tensor(float(v), dtype=torch.float64)
           for k, v in params.items()}
    with torch.no_grad():
        return fwd(p_t).numpy()


def _rmse_by(label, mask, log_L, pred):
    if not mask.any():
        return None
    r = log_L[mask] - pred[mask]
    return dict(label=label, n=int(mask.sum()),
                rmse=float(np.sqrt(np.mean(r ** 2))),
                max_abs=float(np.max(np.abs(r))),
                mean=float(np.mean(r)))


def report_table(data, params, label):
    log_L = np.log(data["L"])
    pred = predict_log_L(params, data["N"], data["D"],
                          data["Dp"], data["source"])
    print(f"\n  {label}  ({len(log_L)} points)")
    print(f"    {'subset':<20s}  {'n':>4s}  {'RMSE':>8s}  "
          f"{'max|Δ|':>8s}  {'mean':>8s}")
    for tag in sorted(set(data["tags"]), key=lambda t: SIZES[t][0]):
        m = data["tags"] == tag
        for src_id, src_name in [(SOURCE_NONE, "1ep"),
                                  (SOURCE_REPEAT, "rep"),
                                  (SOURCE_PARA, "para")]:
            mm = m & (data["source"] == src_id)
            if not mm.any():
                continue
            r = _rmse_by(f"{tag} {src_name}", mm, log_L, pred)
            print(f"    {tag} {src_name:<14s}  {r['n']:>4d}  "
                  f"{r['rmse']:>8.4f}  {r['max_abs']:>8.4f}  "
                  f"{r['mean']:+8.4f}")
    # totals
    full = _rmse_by("ALL", np.ones(len(log_L), dtype=bool), log_L, pred)
    print(f"    {'TOTAL':<20s}  {full['n']:>4d}  "
          f"{full['rmse']:>8.4f}  {full['max_abs']:>8.4f}  "
          f"{full['mean']:+8.4f}")
    return pred


def fit_triple_on_subset(data, k_canonical=15, anchored=True):
    """Re-fit the triple model on `data`. If `anchored`, warm-start from
    the §6 TRIPLE k=15 anchors (TRIPLE_ANCHOR) — necessary when fitting
    on a 2-N subset, where α / (E, A) is not separately identifiable
    from the data alone. Otherwise run the full §6 pipeline."""
    if anchored:
        print(f"\n[Anchored fit] warm-start from §6 TRIPLE k=15 anchors,"
              f" optimize on n={len(data['L'])} small-N points...")
        anchor = dict(
            e=float(np.log(TRIPLE_ANCHOR["E"])),
            a=float(np.log(TRIPLE_ANCHOR["A"])),
            b=float(np.log(TRIPLE_ANCHOR["B"])),
            alpha=TRIPLE_ANCHOR["alpha"],
            beta=TRIPLE_ANCHOR["beta"],
            log_K_rep=TRIPLE_ANCHOR["log_K_rep"],
            rho_rep=TRIPLE_ANCHOR["rho_rep"],
            sigma_rep=TRIPLE_ANCHOR["sigma_rep"],
            log_K_para=TRIPLE_ANCHOR["log_K_para"],
            rho_para=TRIPLE_ANCHOR["rho_para"],
            sigma_para=TRIPLE_ANCHOR["sigma_para"],
        )
        fwd = make_triple_forward(data["N"], data["D"],
                                    data["Dp"], data["source"])
        log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
        res = fit_lse(fwd, log_L,
                       [{k: float(v) for k, v in anchor.items()}],
                       delta=DELTA, verbose=False)
        _print_params("(anchored fit before residual drop)", res["params"])
        sweep = topk_drop_sweep(data, res["params"],
                                  k_values=[0, 5, 10, 15, 20, 25])
    else:
        print(f"\n[Full pipeline] stage 1 + stage 2 + drop on n={len(data['L'])}...")
        res1 = stage1_rep_only(data); p1 = res1["params"]
        _print_params("(stage 1)", p1)
        res2 = stage2_triple(data, p1); p2 = res2["params"]
        _print_params("(stage 2)", p2)
        sweep = topk_drop_sweep(data, p2, k_values=[0, 5, 10, 15, 20, 25])
    rc = sweep[k_canonical]
    _print_params(f"(canonical k={k_canonical})", rc["params"])
    return rc["params"]


def main():
    print("=" * 96)
    print("Extrapolation test: fit on N ≤ 30M, predict held-out 190M / 370M / 600M")
    print("=" * 96)

    full = collect_pooled_triple(scale_min_para=0.5)
    is_small = full["N"] <= SMALL_N_CUT
    is_held = ~is_small  # 60M and above
    is_target = np.isin(full["tags"], HELDOUT_TAGS)

    print(f"\nSmall-N fit set (N ≤ 30M): n={int(is_small.sum())}")
    for tag in ["14m", "30m"]:
        m = full["tags"] == tag
        if not m.any(): continue
        print(f"  {tag}: 1ep={int(((full['source']==SOURCE_NONE)&m).sum())}, "
              f"rep={int(((full['source']==SOURCE_REPEAT)&m).sum())}, "
              f"para={int(((full['source']==SOURCE_PARA)&m).sum())}")
    print(f"\nHeld-out targets {{190M, 370M, 600M}}: n={int(is_target.sum())}")
    for tag in HELDOUT_TAGS:
        m = full["tags"] == tag
        if not m.any(): continue
        print(f"  {tag}: 1ep={int(((full['source']==SOURCE_NONE)&m).sum())}, "
              f"rep={int(((full['source']==SOURCE_REPEAT)&m).sum())}, "
              f"para={int(((full['source']==SOURCE_PARA)&m).sum())}")

    fit_data = filter_data(full, is_small)
    held_data = filter_data(full, is_target)

    # ── Fit on small-N only ────────────────────────────────────────
    params_small = fit_triple_on_subset(fit_data, k_canonical=15)

    # ── Compare with full-data §6 anchors (re-fit for sanity check) ─
    print("\n" + "=" * 96)
    print("In-sample (small-N fit, k=15) RMSE on the fit set:")
    print("=" * 96)
    pred_fit = report_table(fit_data, params_small, "small-N in-sample")

    print("\n" + "=" * 96)
    print("Out-of-sample RMSE on held-out N ∈ {190M, 370M, 600M}:")
    print("=" * 96)
    pred_held = report_table(held_data, params_small, "held-out predictions")

    # ── Parity plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    ax_par, ax_res, ax_dn = axes
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(HELDOUT_TAGS)))
    color_of = {s: cmap[i] for i, s in enumerate(HELDOUT_TAGS)}
    SRC_MARKER = {SOURCE_NONE: "o", SOURCE_REPEAT: "x", SOURCE_PARA: "s"}
    SRC_NAME   = {SOURCE_NONE: "1ep", SOURCE_REPEAT: "rep", SOURCE_PARA: "para"}

    log_L_held = np.log(held_data["L"])
    pred_log_held = predict_log_L(params_small, held_data["N"],
                                    held_data["D"], held_data["Dp"],
                                    held_data["source"])
    L_pred_held = np.exp(pred_log_held)
    resid = log_L_held - pred_log_held

    for tag in HELDOUT_TAGS:
        m = held_data["tags"] == tag
        if not m.any(): continue
        c = color_of[tag]
        for src_id in [SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA]:
            mm = m & (held_data["source"] == src_id)
            if not mm.any(): continue
            mk = SRC_MARKER[src_id]
            edge = "k" if mk != "x" else "none"
            ax_par.scatter(L_pred_held[mm], held_data["L"][mm], s=60,
                           color=c, edgecolors=edge, linewidths=0.4,
                           marker=mk, alpha=0.85,
                           label=(f"{tag} {SRC_NAME[src_id]}"
                                  if src_id == SOURCE_NONE else None))
            ax_res.scatter(held_data["D"][mm] + held_data["Dp"][mm],
                           resid[mm], s=55, color=c,
                           edgecolors=edge, linewidths=0.4, marker=mk)
            ax_dn.scatter(held_data["D"][mm] / held_data["N"][mm],
                          resid[mm], s=55, color=c,
                          edgecolors=edge, linewidths=0.4, marker=mk)
    lo = min(L_pred_held.min(), held_data["L"].min()) * 0.95
    hi = max(L_pred_held.max(), held_data["L"].max()) * 1.05
    ax_par.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6)
    ax_par.set_xlabel("Predicted L  (from N ≤ 30M fit)")
    ax_par.set_ylabel("Observed L  (held-out 190M / 370M / 600M)")
    rmse_held = float(np.sqrt(np.mean(resid ** 2)))
    ax_par.set_title(f"Parity — held-out predictions  "
                     f"(RMSE = {rmse_held:.4f}, n = {len(resid)})",
                     fontsize=12)
    ax_par.legend(loc="lower right", fontsize=9, ncol=2)
    ax_par.grid(alpha=0.3)

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{x/1e9:.1f}B" if x >= 1e9
        else f"{x/1e6:.0f}M"))
    ax_res.set_xlabel(r"Effective tokens $D + D'$")
    ax_res.set_ylabel(r"residual ($\log L$)")
    ax_res.set_title("Residuals vs. effective tokens")
    ax_res.grid(alpha=0.3)

    ax_dn.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_dn.set_xscale("log", base=2)
    ax_dn.set_xlabel(r"$D/N$  (Chinchilla scale × 20)")
    ax_dn.set_ylabel(r"residual ($\log L$)")
    ax_dn.set_title("Residuals vs. $D/N$  (extrapolation regime)")
    ax_dn.grid(alpha=0.3)

    fig.suptitle(
        r"Extrapolation: triple fit on $N \leq30$M predicts $N \in \{190, 370, 600\}$M  "
        r"(○ 1ep, × repeat, □ para)",
        fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fit_triple_extrapolate.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── Side-by-side numerical comparison ──────────────────────────
    print("\n" + "=" * 96)
    print("Summary: in-sample vs out-of-sample RMSE (log L)")
    print("=" * 96)
    log_L_fit = np.log(fit_data["L"])
    pred_log_fit = predict_log_L(params_small, fit_data["N"],
                                   fit_data["D"], fit_data["Dp"],
                                   fit_data["source"])
    resid_fit = log_L_fit - pred_log_fit
    rmse_fit_full = float(np.sqrt(np.mean(resid_fit ** 2)))
    print(f"  In-sample (N ≤ 30M, n={len(log_L_fit)}): "
          f"RMSE = {rmse_fit_full:.4f}")
    for tag in HELDOUT_TAGS:
        m = held_data["tags"] == tag
        if not m.any(): continue
        for src_id, src_name in [(SOURCE_NONE, "1ep"),
                                  (SOURCE_REPEAT, "rep"),
                                  (SOURCE_PARA, "para")]:
            mm = m & (held_data["source"] == src_id)
            if not mm.any(): continue
            r = log_L_held[mm] - pred_log_held[mm]
            print(f"  {tag} {src_name:<6s}  n={int(mm.sum()):>3d}  "
                  f"RMSE={float(np.sqrt(np.mean(r ** 2))):.4f}  "
                  f"max|Δ|={float(np.max(np.abs(r))):.4f}")
    print(f"  Held-out total (n={int(is_target.sum())}): RMSE = {rmse_held:.4f}")

    return params_small, rmse_fit_full, rmse_held


if __name__ == "__main__":
    main()
