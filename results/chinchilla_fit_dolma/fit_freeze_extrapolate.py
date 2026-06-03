"""
Cross-validation / extrapolation test for the freeze pipeline.

Refit the freeze pipeline (Stage 1 = Chinchilla + η_rep on 1-ep + rep,
Stage 2 = η_para fitted on top of frozen Stage 1) using **only N ≤ 30M
runs**, then predict held-out N ∈ {190M, 370M, 600M} validation losses
across all D' regimes.

Differs from fit_triple_extrapolate.py (which is for the §6 joint
pipeline) — here Stage 1's Chinchilla + η_rep is pinned by the small-N
rep+1ep data, and η_para is fit on top in a 3-parameter sub-fit.

Note: the freeze pipeline normally excludes 14M from Stage 1 input
(since 14M's small-scale 1-ep / rep points have high residuals).  For
the small-N extrapolation we *include* 14M in Stage 1 — otherwise
Stage 1 would have only 30M (one size) and α becomes unidentifiable.

Output:
  fit_freeze_extrapolate.pdf — parity / residual diagnostic.
"""

import glob
import os
import sys
from typing import Dict

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

from data import (OVERFIT_EXCLUDE, SIZES,  # noqa: E402
                  extract_1epoch, extract_multi_epoch,
                  extract_paraphrase, load_with_para)
from fit_joint_freeze import (DELTA, _summary, pick_canonical_k,  # noqa: E402
                                stage1_rep_only, stage2_para_only,
                                topk_drop_sweep)
from fit_joint_triple import (SOURCE_NONE, SOURCE_PARA, SOURCE_REPEAT,  # noqa: E402
                                make_triple_forward)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SMALL_N_CUT = 30e6 + 1.0
HELDOUT_TAGS = ("190m", "370m", "600m")

# Reference: freeze-fit on FULL data, k=20 (collaborator's published anchors)
REF_FREEZE_FULL_K20 = dict(
    E=1.245, A=391, B=4068, alpha=0.330, beta=0.357,
    log_K_rep=14.14, rho_rep=-0.738, sigma_rep=-0.546,
    log_K_para=42.34, rho_para=-2.43, sigma_para=-1.74,
)


def collect_pooled_with_size_cut(n_max=None):
    """1-ep + rep + para across sizes, optional N≤n_max filter.

    Notes:
      • Unlike fit_joint_freeze.collect_pooled, we *include* 14M in 1ep+rep
        — small-N extrapolation needs at least 2 sizes for Stage 1 to
        identify α.
      • 14M paraphrase is always kept.
    """
    tags, Ns, Ds, Dps, Ls, src = [], [], [], [], [], []
    for size in SIZES:
        N, datasets, parap = load_with_para(size)
        if n_max is not None and N > n_max:
            continue
        # 1-ep
        _, D1, L1 = extract_1epoch(datasets, N, scale_min=0.0)
        for d, l in zip(D1, L1):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(0.0)
            Ls.append(l); src.append(SOURCE_NONE)
        # rep
        _, Dm, _, Dpm, Lm, _ = extract_multi_epoch(
            datasets, N, scale_min=0.0,
            exclude_overfit=OVERFIT_EXCLUDE.get(size, set()))
        for d, dp, l in zip(Dm, Dpm, Lm):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
            Ls.append(l); src.append(SOURCE_REPEAT)
        # para
        if parap:
            _, Dp_, _, Dpp, Lp, _ = extract_paraphrase(
                datasets, parap, N, scale_min=0.0)
            for d, dp, l in zip(Dp_, Dpp, Lp):
                tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
                Ls.append(l); src.append(SOURCE_PARA)
    return dict(
        tags=np.array(tags), N=np.array(Ns, dtype=np.float64),
        D=np.array(Ds, dtype=np.float64), Dp=np.array(Dps, dtype=np.float64),
        L=np.array(Ls, dtype=np.float64), source=np.array(src, dtype=np.int64),
    )


def predict_log_L(params, N, D, Dp, source):
    fwd = make_triple_forward(N, D, Dp, source)
    p_t = {k: torch.tensor(float(v), dtype=torch.float64)
           for k, v in params.items()}
    with torch.no_grad():
        return fwd(p_t).numpy()


def report_table(data, params, label):
    log_L = np.log(data["L"])
    pred = predict_log_L(params, data["N"], data["D"],
                          data["Dp"], data["source"])
    resid = log_L - pred
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
            r = resid[mm]
            print(f"    {tag} {src_name:<14s}  {int(mm.sum()):>4d}  "
                  f"{float(np.sqrt(np.mean(r**2))):>8.4f}  "
                  f"{float(np.max(np.abs(r))):>8.4f}  "
                  f"{float(np.mean(r)):+8.4f}")
    full = resid
    print(f"    {'TOTAL':<20s}  {len(full):>4d}  "
          f"{float(np.sqrt(np.mean(full**2))):>8.4f}  "
          f"{float(np.max(np.abs(full))):>8.4f}  "
          f"{float(np.mean(full)):+8.4f}")
    return pred


def filter_data(data, mask):
    return {k: v[mask] for k, v in data.items()}


def fit_freeze_on_subset(fit_data, k_canonical=20, anchored=True):
    """Run the freeze pipeline (Stage 1 → Stage 2 → drop sweep) on
    `fit_data`.

    With `anchored=True` (default), Stage 1 is warm-started from the
    full-data freeze anchors to avoid the degenerate (α≈1.6, A≈10^{11})
    basin that L-BFGS finds when only 2 N-values are present in
    Stage 1's input.  This makes the experiment test whether the
    functional form extrapolates, not whether 2 N-points constrain α.
    """
    from fit_lse import fit_lse  # local import
    n_rep1ep = int((fit_data['source']!=SOURCE_PARA).sum())
    print(f"\n[Stage 1] Chinchilla + η_rep on 1ep+rep (n={n_rep1ep})  "
          f"{'[anchored to full-data freeze]' if anchored else '[free fit]'}...")
    if anchored:
        # Warm-start from the full-data freeze fit (k=20 anchors)
        anchor = dict(
            e=float(np.log(REF_FREEZE_FULL_K20["E"])),
            a=float(np.log(REF_FREEZE_FULL_K20["A"])),
            b=float(np.log(REF_FREEZE_FULL_K20["B"])),
            alpha=REF_FREEZE_FULL_K20["alpha"],
            beta=REF_FREEZE_FULL_K20["beta"],
            log_K_rep=REF_FREEZE_FULL_K20["log_K_rep"],
            rho_rep=REF_FREEZE_FULL_K20["rho_rep"],
            sigma_rep=REF_FREEZE_FULL_K20["sigma_rep"],
        )
        from fit_joint_triple import make_forward_rep_only
        keep = fit_data["source"] != SOURCE_PARA
        fwd = make_forward_rep_only(
            fit_data["N"][keep], fit_data["D"][keep], fit_data["Dp"][keep],
            is_multi_arr=(fit_data["source"][keep] == SOURCE_REPEAT))
        log_L = torch.tensor(np.log(fit_data["L"][keep]), dtype=torch.float64)
        res1 = fit_lse(fwd, log_L,
                       [{k: float(v) for k, v in anchor.items()}],
                       delta=DELTA, verbose=False)
    else:
        res1 = stage1_rep_only(fit_data)
    p1 = res1["params"]
    print(f"  E={np.exp(p1['e']):.4f}  A={np.exp(p1['a']):.2f}  "
          f"B={np.exp(p1['b']):.2f}  α={p1['alpha']:.4f}  "
          f"β={p1['beta']:.4f}")
    print(f"  log K_rep={p1['log_K_rep']:.2f}  ρ_rep={p1['rho_rep']:+.3f}  "
          f"σ_rep={p1['sigma_rep']:+.3f}")

    fixed_keys = ["e", "a", "b", "alpha", "beta",
                  "log_K_rep", "rho_rep", "sigma_rep"]
    fixed_params = {k: p1[k] for k in fixed_keys}

    print(f"\n[Stage 2] η_para fitted on top of frozen Stage 1 "
          f"(para n={int((fit_data['source']==SOURCE_PARA).sum())})...")
    res2 = stage2_para_only(fit_data, fixed_params)
    para_init = res2["params"]
    print(f"  log K_para={para_init['log_K_para']:.2f}  "
          f"ρ_para={para_init['rho_para']:+.3f}  "
          f"σ_para={para_init['sigma_para']:+.3f}")

    K_VALUES = [0, 5, 10, 15, 20, 25, 30]
    print(f"\n[Stage 3] Iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(fit_data, para_init, fixed_params, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  | "
          f"{'lK_par':>7s} {'ρ_par':>7s} {'σ_par':>7s} | "
          f"{'1ep':>5s}  {'rep':>5s}  {'par':>5s}")
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(fit_data, r["params"], r["pred_full"], r["keep"])
        p = r["params"]
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  | "
              f"{p['log_K_para']:>7.2f} {p['rho_para']:>+7.3f} "
              f"{p['sigma_para']:>+7.3f} | "
              f"{s['rmse_1ep']:>5.3f}  {s['rmse_rep']:>5.3f}  "
              f"{s['rmse_para']:>5.3f}")

    rc = sweep[k_canonical]
    sc = _summary(fit_data, rc["params"], rc["pred_full"], rc["keep"])
    print(f"\n  Reporting canonical k={k_canonical}:")
    print(f"    fit RMSE — 1ep:{sc['rmse_1ep']:.4f}  rep:{sc['rmse_rep']:.4f}  "
          f"para:{sc['rmse_para']:.4f}  |  kept:{sc['rmse_kept']:.4f}")
    return rc["params"]


def main():
    print("=" * 96)
    print("Freeze-pipeline cross-validation: fit on N ≤ 30M, predict held-out 190M/370M/600M")
    print("=" * 96)

    full = collect_pooled_with_size_cut(n_max=None)
    is_small = full["N"] <= SMALL_N_CUT
    is_target = np.isin(full["tags"], HELDOUT_TAGS)
    print(f"\nFull pool n={len(full['L'])}  "
          f"(1ep={int((full['source']==SOURCE_NONE).sum())}, "
          f"rep={int((full['source']==SOURCE_REPEAT).sum())}, "
          f"para={int((full['source']==SOURCE_PARA).sum())})")
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

    params_small = fit_freeze_on_subset(fit_data, k_canonical=20)

    print("\n" + "=" * 96)
    print("In-sample (small-N freeze fit, k=20) RMSE on the fit set:")
    print("=" * 96)
    pred_fit = report_table(fit_data, params_small, "small-N in-sample")

    print("\n" + "=" * 96)
    print("Out-of-sample RMSE on held-out N ∈ {190M, 370M, 600M}:")
    print("=" * 96)
    pred_held = report_table(held_data, params_small, "held-out predictions")

    # ── Plot ──────────────────────────────────────────────────────
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
    rmse_held = float(np.sqrt(np.mean(resid ** 2)))

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
    ax_par.set_xlabel("Predicted L  (from N ≤ 30M freeze fit)")
    ax_par.set_ylabel("Observed L  (held-out 190M / 370M / 600M)")
    ax_par.set_title(f"Parity — held-out predictions  "
                     f"(RMSE = {rmse_held:.4f}, n = {len(resid)})",
                     fontsize=12)
    ax_par.legend(loc="lower right", fontsize=9, ncol=2)
    ax_par.grid(alpha=0.3)

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M"))
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
        r"Freeze-pipeline extrapolation: fit on $N \leq 30$M predicts "
        r"$N \in \{190, 370, 600\}$M  (○ 1ep, × repeat, □ para)",
        fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fit_freeze_extrapolate.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("Summary: freeze pipeline, in-sample vs out-of-sample RMSE (log L)")
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
    print(f"  Held-out total (n={int(is_target.sum())}): "
          f"RMSE = {rmse_held:.4f}")

    # ── Compare params with the FULL freeze fit ───────────────────
    print("\n" + "=" * 96)
    print("Param comparison: small-N freeze fit  vs  full-data freeze (k=20)")
    print("=" * 96)
    pc = params_small
    print(f"  {'param':<14s}  {'full-data (ref)':>18s}  "
          f"{'small-N (this fit)':>20s}  {'Δ':>10s}")
    rows = [
        ("E",          REF_FREEZE_FULL_K20["E"],          float(np.exp(pc["e"]))),
        ("A",          REF_FREEZE_FULL_K20["A"],          float(np.exp(pc["a"]))),
        ("B",          REF_FREEZE_FULL_K20["B"],          float(np.exp(pc["b"]))),
        ("alpha",      REF_FREEZE_FULL_K20["alpha"],      pc["alpha"]),
        ("beta",       REF_FREEZE_FULL_K20["beta"],       pc["beta"]),
        ("log_K_rep",  REF_FREEZE_FULL_K20["log_K_rep"],  pc["log_K_rep"]),
        ("rho_rep",    REF_FREEZE_FULL_K20["rho_rep"],    pc["rho_rep"]),
        ("sigma_rep",  REF_FREEZE_FULL_K20["sigma_rep"],  pc["sigma_rep"]),
        ("log_K_para", REF_FREEZE_FULL_K20["log_K_para"], pc["log_K_para"]),
        ("rho_para",   REF_FREEZE_FULL_K20["rho_para"],   pc["rho_para"]),
        ("sigma_para", REF_FREEZE_FULL_K20["sigma_para"], pc["sigma_para"]),
    ]
    for name, ref, val in rows:
        delta = val - ref
        if name in ("E", "A", "B"):
            print(f"  {name:<14s}  {ref:>18.4g}  {val:>20.4g}  {delta:>+10.4g}")
        else:
            print(f"  {name:<14s}  {ref:>+18.4f}  {val:>+20.4f}  {delta:>+10.4f}")

    return params_small, rmse_fit_full, rmse_held


if __name__ == "__main__":
    main()
