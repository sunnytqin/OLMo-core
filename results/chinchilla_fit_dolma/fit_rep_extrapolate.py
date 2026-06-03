"""
Cross-validation / extrapolation test for the rep+1ep model (no
paraphrase): fit on N ≤ 30M, predict held-out N ∈ {190M, 370M, 600M}.

Mirrors fit_triple_extrapolate.py and fit_freeze_extrapolate.py but on
the 9-parameter sub-model from writeup_final §2:

    L = E + A/N^α + B / (D + η_rep · D')^β
    log R*_rep = log K_rep + ρ_rep · log(D/N) + σ_rep · log N

Stage 1 is warm-started from writeup_final's k=15 anchors so the
2-N-value training set doesn't blow up (E, A, α).
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

from data import (OVERFIT_EXCLUDE, SIZES,  # noqa: E402
                  extract_1epoch, extract_multi_epoch, load)
from fit_joint_triple import (DELTA, SOURCE_NONE, SOURCE_REPEAT,  # noqa: E402
                                make_forward_rep_only)
from fit_lse import fit_lse  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SMALL_N_CUT = 30e6 + 1.0
HELDOUT_TAGS = ("190m", "370m", "600m")

# writeup_final §2 headline anchors (one-shot rep+1ep, k=15)
REF_REP_FULL_K15 = dict(
    E=0.050, A=31.5, B=16539, alpha=0.137, beta=0.436,
    log_K_rep=10.32, rho_rep=-0.270, sigma_rep=-0.388,
)


def collect_rep_pool(n_max=None):
    """Return 1-ep + rep pooled across sizes; optional N≤n_max filter."""
    tags, Ns, Ds, Dps, Ls, src = [], [], [], [], [], []
    for size in SIZES:
        N, datasets = load(size)
        if n_max is not None and N > n_max:
            continue
        _, D1, L1 = extract_1epoch(datasets, N, scale_min=0.0)
        for d, l in zip(D1, L1):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(0.0)
            Ls.append(l); src.append(SOURCE_NONE)
        _, Dm, _, Dpm, Lm, _ = extract_multi_epoch(
            datasets, N, scale_min=0.0,
            exclude_overfit=OVERFIT_EXCLUDE.get(size, set()))
        for d, dp, l in zip(Dm, Dpm, Lm):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
            Ls.append(l); src.append(SOURCE_REPEAT)
    return dict(
        tags=np.array(tags), N=np.array(Ns, dtype=np.float64),
        D=np.array(Ds, dtype=np.float64), Dp=np.array(Dps, dtype=np.float64),
        L=np.array(Ls, dtype=np.float64), source=np.array(src, dtype=np.int64),
    )


def predict_log_L(params, data):
    fwd = make_forward_rep_only(
        data["N"], data["D"], data["Dp"],
        is_multi_arr=(data["source"] == SOURCE_REPEAT))
    p_t = {k: torch.tensor(float(v), dtype=torch.float64)
           for k, v in params.items()}
    with torch.no_grad():
        return fwd(p_t).numpy()


def filter_data(data, mask):
    return {k: v[mask] for k, v in data.items()}


def fit_rep_iterative(fit_data, anchor, k_values, delta=DELTA):
    """Single-init warm-start from `anchor` then iterative residual drop."""
    fwd_full = make_forward_rep_only(
        fit_data["N"], fit_data["D"], fit_data["Dp"],
        is_multi_arr=(fit_data["source"] == SOURCE_REPEAT))
    log_L_full = torch.tensor(np.log(fit_data["L"]), dtype=torch.float64)
    res = fit_lse(fwd_full, log_L_full,
                   [{k: float(v) for k, v in anchor.items()}],
                   delta=delta, verbose=False)
    last = res["params"]
    keep = np.ones(len(fit_data["L"]), dtype=bool)
    cumulative = []
    out = {}
    for k in sorted(set(k_values)):
        while len(cumulative) < k:
            grid = [{kk: float(vv) for kk, vv in last.items()}]
            fwd = make_forward_rep_only(
                fit_data["N"][keep], fit_data["D"][keep], fit_data["Dp"][keep],
                is_multi_arr=(fit_data["source"][keep] == SOURCE_REPEAT))
            log_L = torch.tensor(np.log(fit_data["L"][keep]), dtype=torch.float64)
            res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
            with torch.no_grad():
                p_t = {kk: torch.tensor(v, dtype=torch.float64)
                       for kk, v in res["params"].items()}
                pred_full = fwd_full(p_t).numpy()
            resid = np.log(fit_data["L"]) - pred_full
            ar = np.abs(resid); ar[~keep] = -1.0
            worst = int(np.argmax(ar))
            cumulative.append(worst); keep[worst] = False
            last = res["params"]
        # Final fit at this k
        grid = [{kk: float(vv) for kk, vv in last.items()}]
        fwd = make_forward_rep_only(
            fit_data["N"][keep], fit_data["D"][keep], fit_data["Dp"][keep],
            is_multi_arr=(fit_data["source"][keep] == SOURCE_REPEAT))
        log_L = torch.tensor(np.log(fit_data["L"][keep]), dtype=torch.float64)
        res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
        last = res["params"]
        with torch.no_grad():
            p_t = {kk: torch.tensor(v, dtype=torch.float64)
                   for kk, v in res["params"].items()}
            pred_full = fwd_full(p_t).numpy()
        out[k] = dict(params=res["params"],
                       pred_full=pred_full, keep=keep.copy(),
                       dropped=list(cumulative))
    return out


def report_table(data, params, label):
    log_L = np.log(data["L"])
    pred = predict_log_L(params, data)
    resid = log_L - pred
    print(f"\n  {label}  ({len(log_L)} points)")
    print(f"    {'subset':<20s}  {'n':>4s}  {'RMSE':>8s}  "
          f"{'max|Δ|':>8s}  {'mean':>8s}")
    for tag in sorted(set(data["tags"]), key=lambda t: SIZES[t][0]):
        m = data["tags"] == tag
        for src_id, src_name in [(SOURCE_NONE, "1ep"), (SOURCE_REPEAT, "rep")]:
            mm = m & (data["source"] == src_id)
            if not mm.any(): continue
            r = resid[mm]
            print(f"    {tag} {src_name:<14s}  {int(mm.sum()):>4d}  "
                  f"{float(np.sqrt(np.mean(r**2))):>8.4f}  "
                  f"{float(np.max(np.abs(r))):>8.4f}  "
                  f"{float(np.mean(r)):+8.4f}")
    print(f"    {'TOTAL':<20s}  {len(resid):>4d}  "
          f"{float(np.sqrt(np.mean(resid**2))):>8.4f}  "
          f"{float(np.max(np.abs(resid))):>8.4f}  "
          f"{float(np.mean(resid)):+8.4f}")
    return pred


def main():
    print("=" * 96)
    print("Rep+1ep cross-validation: fit on N ≤ 30M, predict held-out 190M/370M/600M")
    print("=" * 96)

    full = collect_rep_pool()
    is_small = full["N"] <= SMALL_N_CUT
    is_target = np.isin(full["tags"], HELDOUT_TAGS)
    print(f"\nFull pool n={len(full['L'])}  "
          f"(1ep={int((full['source']==SOURCE_NONE).sum())}, "
          f"rep={int((full['source']==SOURCE_REPEAT).sum())})")
    print(f"\nSmall-N fit set (N ≤ 30M): n={int(is_small.sum())}")
    for tag in ["14m", "30m"]:
        m = full["tags"] == tag
        if not m.any(): continue
        print(f"  {tag}: 1ep={int(((full['source']==SOURCE_NONE)&m).sum())}, "
              f"rep={int(((full['source']==SOURCE_REPEAT)&m).sum())}")
    print(f"\nHeld-out targets {{190M, 370M, 600M}}: n={int(is_target.sum())}")
    for tag in HELDOUT_TAGS:
        m = full["tags"] == tag
        if not m.any(): continue
        print(f"  {tag}: 1ep={int(((full['source']==SOURCE_NONE)&m).sum())}, "
              f"rep={int(((full['source']==SOURCE_REPEAT)&m).sum())}")

    fit_data = filter_data(full, is_small)
    held_data = filter_data(full, is_target)

    # ── Fit on small-N (anchored) ──────────────────────────────────
    anchor = dict(
        e=float(np.log(REF_REP_FULL_K15["E"])),
        a=float(np.log(REF_REP_FULL_K15["A"])),
        b=float(np.log(REF_REP_FULL_K15["B"])),
        alpha=REF_REP_FULL_K15["alpha"],
        beta=REF_REP_FULL_K15["beta"],
        log_K_rep=REF_REP_FULL_K15["log_K_rep"],
        rho_rep=REF_REP_FULL_K15["rho_rep"],
        sigma_rep=REF_REP_FULL_K15["sigma_rep"],
    )
    print("\n[Fit] anchored to writeup_final §2 k=15 anchors, optimize on "
          f"N ≤ 30M (n={len(fit_data['L'])})...")
    K_VALUES = [0, 5, 10, 15, 20, 25]
    sweep = fit_rep_iterative(fit_data, anchor, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  {'E':>7s}  {'A':>7s}  "
          f"{'B':>9s}  {'α':>6s}  {'β':>6s}  | "
          f"{'lK_r':>6s} {'ρ_r':>6s} {'σ_r':>6s}")
    for k in K_VALUES:
        r = sweep[k]
        p = r["params"]
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  "
              f"{np.exp(p['e']):>7.3f}  {np.exp(p['a']):>7.1f}  "
              f"{np.exp(p['b']):>9.0f}  {p['alpha']:>6.3f}  "
              f"{p['beta']:>6.3f}  | "
              f"{p['log_K_rep']:>6.2f} {p['rho_rep']:>+6.2f} "
              f"{p['sigma_rep']:>+6.2f}")

    canonical = 15
    rc = sweep[canonical]
    pc = rc["params"]
    print(f"\n  Reporting canonical k={canonical}:")
    print(f"    E={np.exp(pc['e']):.4f}  A={np.exp(pc['a']):.2f}  "
          f"B={np.exp(pc['b']):.2f}  α={pc['alpha']:.4f}  "
          f"β={pc['beta']:.4f}")
    print(f"    log K_rep={pc['log_K_rep']:.2f}  ρ_rep={pc['rho_rep']:+.3f}  "
          f"σ_rep={pc['sigma_rep']:+.3f}")

    print("\n" + "=" * 96)
    print("In-sample (small-N rep+1ep, k=15) RMSE on the fit set:")
    print("=" * 96)
    report_table(fit_data, pc, "small-N in-sample")

    print("\n" + "=" * 96)
    print("Out-of-sample RMSE on held-out N ∈ {190M, 370M, 600M}:")
    print("=" * 96)
    report_table(held_data, pc, "held-out predictions")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    ax_par, ax_res, ax_dn = axes
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(HELDOUT_TAGS)))
    color_of = {s: cmap[i] for i, s in enumerate(HELDOUT_TAGS)}
    SRC_MARKER = {SOURCE_NONE: "o", SOURCE_REPEAT: "x"}
    SRC_NAME   = {SOURCE_NONE: "1ep", SOURCE_REPEAT: "rep"}

    log_L_held = np.log(held_data["L"])
    pred_log_held = predict_log_L(pc, held_data)
    L_pred_held = np.exp(pred_log_held)
    resid = log_L_held - pred_log_held
    rmse_held = float(np.sqrt(np.mean(resid ** 2)))

    for tag in HELDOUT_TAGS:
        m = held_data["tags"] == tag
        if not m.any(): continue
        c = color_of[tag]
        for src_id in [SOURCE_NONE, SOURCE_REPEAT]:
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
    ax_par.set_xlabel("Predicted L  (from N ≤ 30M rep+1ep fit)")
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
    ax_dn.set_title("Residuals vs. $D/N$")
    ax_dn.grid(alpha=0.3)

    fig.suptitle(
        r"Rep+1ep extrapolation: fit on $N \leq 30$M predicts "
        r"$N \in \{190, 370, 600\}$M  (○ 1ep, × repeat)",
        fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fit_rep_extrapolate.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("Summary: rep+1ep pipeline, in-sample vs out-of-sample RMSE (log L)")
    print("=" * 96)
    log_L_fit = np.log(fit_data["L"])
    pred_fit = predict_log_L(pc, fit_data)
    resid_fit = log_L_fit - pred_fit
    rmse_fit_full = float(np.sqrt(np.mean(resid_fit ** 2)))
    print(f"  In-sample (N ≤ 30M, n={len(log_L_fit)}): "
          f"RMSE = {rmse_fit_full:.4f}")
    for tag in HELDOUT_TAGS:
        m = held_data["tags"] == tag
        if not m.any(): continue
        for src_id, src_name in [(SOURCE_NONE, "1ep"),
                                  (SOURCE_REPEAT, "rep")]:
            mm = m & (held_data["source"] == src_id)
            if not mm.any(): continue
            r = log_L_held[mm] - pred_log_held[mm]
            print(f"  {tag} {src_name:<6s}  n={int(mm.sum()):>3d}  "
                  f"RMSE={float(np.sqrt(np.mean(r ** 2))):.4f}  "
                  f"max|Δ|={float(np.max(np.abs(r))):.4f}")
    print(f"  Held-out total (n={int(is_target.sum())}): "
          f"RMSE = {rmse_held:.4f}")

    # ── Side-by-side comparison with full-data anchors ─────────────
    print("\n" + "=" * 96)
    print("Param comparison: small-N rep+1ep fit  vs  full-data writeup_final §2 (k=15)")
    print("=" * 96)
    print(f"  {'param':<14s}  {'full-data ref':>15s}  "
          f"{'small-N (this fit)':>20s}  {'Δ':>10s}")
    rows = [
        ("E",          REF_REP_FULL_K15["E"],          float(np.exp(pc["e"]))),
        ("A",          REF_REP_FULL_K15["A"],          float(np.exp(pc["a"]))),
        ("B",          REF_REP_FULL_K15["B"],          float(np.exp(pc["b"]))),
        ("alpha",      REF_REP_FULL_K15["alpha"],      pc["alpha"]),
        ("beta",       REF_REP_FULL_K15["beta"],       pc["beta"]),
        ("log_K_rep",  REF_REP_FULL_K15["log_K_rep"],  pc["log_K_rep"]),
        ("rho_rep",    REF_REP_FULL_K15["rho_rep"],    pc["rho_rep"]),
        ("sigma_rep",  REF_REP_FULL_K15["sigma_rep"],  pc["sigma_rep"]),
    ]
    for name, ref, val in rows:
        delta = val - ref
        if name in ("E", "A", "B"):
            print(f"  {name:<14s}  {ref:>15.4g}  {val:>20.4g}  {delta:>+10.4g}")
        else:
            print(f"  {name:<14s}  {ref:>+15.4f}  {val:>+20.4f}  {delta:>+10.4f}")

    return pc, rmse_fit_full, rmse_held


if __name__ == "__main__":
    main()
