"""
Two-stage joint fit on 1-epoch + repetition + paraphrase data with Stage 1
parameters frozen ("all1" freeze).

Functional form (shared Chinchilla, separate η for repetition vs paraphrase):

    L(N, D, D'; src) = E + A/N^α + B / (D + η_src(D, D'; N) · D')^β

with src ∈ {none (1-epoch), repeat, para}. η_src has Form-B exp-sat shape:

    η_src(D, D'; N) = R*_src · (1 − e^{−x/R*_src}) / x,    x = D'/D,
    log R*_src = log K_src + ρ_src · log(D/N) + σ_src · log N.

Procedure:
    Stage 1.  Fit 9 params (Chinchilla E, A, B, α, β + η_rep) on
              1-epoch + repetition data only (14M excluded).
              This reproduces fit_joint_all.py.

    Stage 2.  Freeze all 8 Stage 1 parameters and fit the 3 remaining
              η_para parameters (log K_para, ρ_para, σ_para) on the full
              pooled 1-ep + rep + para dataset.

    Stage 3.  Iterative residual-greedy drop sweep with η_para refit at
              every k.  Pick canonical k as first plateau in para-RMSE.

Data choices:
  • 14M excluded from 1-epoch and repetition (Stage 1 input).
  • 14M kept for paraphrase (Stage 2/3 input).
  • OVERFIT_EXCLUDE applied to repetition per data.py.

Output:
    fit_joint_freeze.pdf — diagnostic plot (data/residual/parity).
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
from fit_joint_triple import (SOURCE_NONE, SOURCE_PARA, SOURCE_REPEAT,  # noqa: E402
                                GRID_REP_ONLY, PARA_GRID,
                                fmt_tokens, make_forward_rep_only,
                                make_triple_forward)
from fit_lse import fit_lse  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.1
EXCLUDE_SIZES_1EP_REP = ("14m",)


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def collect_pooled():
    """Pool 1-ep + rep + para across model sizes; 14M excluded from 1ep+rep."""
    tags, Ns, Ds, Dps, Ls, src = [], [], [], [], [], []
    for size in SIZES:
        N, datasets, parap = load_with_para(size)
        skip_1ep_rep = size in EXCLUDE_SIZES_1EP_REP
        if not skip_1ep_rep:
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
        if parap:
            _, Dp_, _, Dpp, Lp, _ = extract_paraphrase(
                datasets, parap, N, scale_min=0.0)
            for d, dp, l in zip(Dp_, Dpp, Lp):
                tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
                Ls.append(l); src.append(SOURCE_PARA)
    return dict(
        tags=np.array(tags),
        N=np.array(Ns, dtype=np.float64),
        D=np.array(Ds, dtype=np.float64),
        Dp=np.array(Dps, dtype=np.float64),
        L=np.array(Ls, dtype=np.float64),
        source=np.array(src, dtype=np.int64),
    )


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────

def _wrap_fixed(forward_fn, fixed_params):
    fixed_t = {k: torch.tensor(float(v), dtype=torch.float64)
               for k, v in fixed_params.items()}

    def fwd(p):
        return forward_fn({**p, **fixed_t})

    return fwd


def stage1_rep_only(data, delta=DELTA):
    """Fit 9-param (Chinchilla + η_rep) on 1ep + rep only."""
    keep = data["source"] != SOURCE_PARA
    fwd = make_forward_rep_only(
        data["N"][keep], data["D"][keep], data["Dp"][keep],
        is_multi_arr=(data["source"][keep] == SOURCE_REPEAT))
    log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
    return fit_lse(fwd, log_L, GRID_REP_ONLY, delta=delta, verbose=False)


def stage2_para_only(data, fixed_params, delta=DELTA):
    """Fit only η_para (3 params); all 8 Stage 1 params held fixed."""
    base = make_triple_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    fwd = _wrap_fixed(base, fixed_params)
    init_grid = list(PARA_GRID)
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    return fit_lse(fwd, log_L, init_grid, delta=delta, verbose=False)


def topk_drop_sweep(data, init_para_params, fixed_params, k_values,
                     delta=DELTA):
    """Iterative residual drop with η_para refit at each step."""
    base_full = make_triple_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    fwd_full = _wrap_fixed(base_full, fixed_params)
    last = dict(init_para_params)
    keep = np.ones(len(data["L"]), dtype=bool)
    cumulative = []
    out = {}
    for k in sorted(set(k_values)):
        while len(cumulative) < k:
            grid = [{kk: float(vv) for kk, vv in last.items()}]
            base = make_triple_forward(
                data["N"][keep], data["D"][keep],
                data["Dp"][keep], data["source"][keep])
            fwd = _wrap_fixed(base, fixed_params)
            log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
            res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
            with torch.no_grad():
                p_t = {kk: torch.tensor(v, dtype=torch.float64)
                       for kk, v in res["params"].items()}
                pred_full = fwd_full(p_t).numpy()
            resid = np.log(data["L"]) - pred_full
            ar = np.abs(resid); ar[~keep] = -1.0
            worst = int(np.argmax(ar))
            cumulative.append(worst); keep[worst] = False
            last = res["params"]
        # Final fit at this k
        grid = [{kk: float(vv) for kk, vv in last.items()}]
        base = make_triple_forward(
            data["N"][keep], data["D"][keep],
            data["Dp"][keep], data["source"][keep])
        fwd = _wrap_fixed(base, fixed_params)
        log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
        res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
        last = res["params"]
        with torch.no_grad():
            p_t = {kk: torch.tensor(v, dtype=torch.float64)
                   for kk, v in res["params"].items()}
            pred_full = fwd_full(p_t).numpy()
        full_params = {**res["params"],
                        **{k: float(v) for k, v in fixed_params.items()}}
        out[k] = dict(params=full_params,
                       pred_full=pred_full,
                       keep=keep.copy(),
                       dropped=list(cumulative))
    return out


def _summary(data, params, pred_full, keep):
    log_L = np.log(data["L"])
    resid = log_L - pred_full
    out = dict(rmse_full=float(np.sqrt(np.mean(resid ** 2))),
               rmse_kept=float(np.sqrt(np.mean(resid[keep] ** 2))))
    for tag, src_id in [("1ep", SOURCE_NONE), ("rep", SOURCE_REPEAT),
                         ("para", SOURCE_PARA)]:
        m = (data["source"] == src_id)
        mk = m & keep
        out[f"n_{tag}"] = int(m.sum())
        out[f"n_{tag}_kept"] = int(mk.sum())
        out[f"rmse_{tag}"] = (float(np.sqrt(np.mean(resid[mk] ** 2)))
                              if mk.any() else float("nan"))
    return out


def pick_canonical_k(sweep, data, k_values, tol=0.001):
    """First k whose para-RMSE plateaus vs the previous k (Δ < tol)."""
    rmses = []
    for k in k_values:
        s = _summary(data, sweep[k]["params"],
                     sweep[k]["pred_full"], sweep[k]["keep"])
        rmses.append(s["rmse_para"])
    for i in range(1, len(rmses)):
        if abs(rmses[i] - rmses[i - 1]) < tol:
            return k_values[i]
    return k_values[-1]


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def plot_diagnostic(data, pred_full, keep, dropped, params, summary, path):
    sizes = sorted(set(data["tags"].tolist()), key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sizes)))
    color_of = {s: cmap[i] for i, s in enumerate(sizes)}

    log_L = np.log(data["L"])
    resid = log_L - pred_full

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_data, ax_res, ax_par = axes

    for tag in sizes:
        m = data["tags"] == tag
        c = color_of[tag]
        m1 = m & (data["source"] == SOURCE_NONE)
        ax_data.scatter(data["D"][m1], data["L"][m1], s=70, color=c,
                        edgecolors="k", linewidths=0.4, marker="o",
                        zorder=4, label=tag)
        mr = m & (data["source"] == SOURCE_REPEAT)
        ax_data.scatter(data["D"][mr] + data["Dp"][mr], data["L"][mr],
                        s=42, color=c, marker="x", alpha=0.7, zorder=3)
        mp = m & (data["source"] == SOURCE_PARA)
        ax_data.scatter(data["D"][mp] + data["Dp"][mp], data["L"][mp],
                        s=42, color=c, edgecolors="k", linewidths=0.5,
                        marker="s", alpha=0.85, zorder=3)
        for src_id, mk_marker in [(SOURCE_NONE, "o"),
                                    (SOURCE_REPEAT, "x"),
                                    (SOURCE_PARA, "s")]:
            mm = m & (data["source"] == src_id)
            if mm.any():
                ax_res.scatter(data["D"][mm], resid[mm], s=40, color=c,
                                edgecolors=("k" if mk_marker != "x"
                                              else "none"),
                                linewidths=0.3, marker=mk_marker)
        ax_par.scatter(np.exp(pred_full[m]), data["L"][m], s=42, color=c,
                       edgecolors="k", linewidths=0.3)

    if len(dropped):
        dmask = np.zeros(len(data["L"]), dtype=bool); dmask[dropped] = True
        ax_data.scatter(data["D"][dmask] + data["Dp"][dmask],
                        data["L"][dmask], s=170, facecolors="none",
                        edgecolors="red", linewidths=1.6, zorder=10,
                        label="dropped")

    ax_data.set_xscale("log", base=2)
    ax_data.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_data.set_xlabel(r"Effective tokens $D + D'$")
    ax_data.set_ylabel("Validation loss")
    ax_data.legend(loc="upper right", ncol=2, fontsize=9)
    ax_data.grid(alpha=0.3)
    p = params
    ax_data.set_title(
        rf"Stage-1-frozen fit: $E={np.exp(p['e']):.2f}$, $A={np.exp(p['a']):.0f}$, "
        rf"$B={np.exp(p['b']):.0f}$, $\alpha={p['alpha']:.2f}$, "
        rf"$\beta={p['beta']:.3f}$" + "\n"
        rf"η_rep (frozen): $\log K={p['log_K_rep']:.1f}$, $\rho={p['rho_rep']:+.2f}$, "
        rf"$\sigma={p['sigma_rep']:+.2f}$  |  "
        rf"η_para (fit): $\log K={p['log_K_para']:.1f}$, $\rho={p['rho_para']:+.2f}$, "
        rf"$\sigma={p['sigma_para']:+.2f}$",
        fontsize=11)

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_res.set_xlabel(r"$D$")
    ax_res.set_ylabel(r"residual ($\log L$)")
    ax_res.set_title(rf"Residuals (kept RMSE: 1ep={summary['rmse_1ep']:.3f}, "
                     rf"rep={summary['rmse_rep']:.3f}, "
                     rf"para={summary['rmse_para']:.3f})")
    ax_res.grid(alpha=0.3)

    lo = min(np.exp(pred_full).min(), data["L"].min()) * 0.95
    hi = max(np.exp(pred_full).max(), data["L"].max()) * 1.05
    ax_par.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6)
    ax_par.set_xlabel("Predicted L")
    ax_par.set_ylabel("Observed L")
    ax_par.set_title("Parity")
    ax_par.grid(alpha=0.3)

    fig.suptitle(
        "Stage-1-frozen joint fit: η_para fitted on top of frozen "
        "Chinchilla + η_rep   (○ 1ep, × repeat, □ para)",
        fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _print_params(label, p):
    print(f"  {label}")
    print(f"    Chinchilla: E={np.exp(p['e']):.4f}  A={np.exp(p['a']):.2f}  "
          f"B={np.exp(p['b']):.2f}  α={p['alpha']:.4f}  β={p['beta']:.4f}")
    if "log_K_rep" in p:
        print(f"    η_rep:      log K={p['log_K_rep']:.2f}  "
              f"ρ={p['rho_rep']:+.3f}  σ={p['sigma_rep']:+.3f}")
    if "log_K_para" in p:
        print(f"    η_para:     log K={p['log_K_para']:.2f}  "
              f"ρ={p['rho_para']:+.3f}  σ={p['sigma_para']:+.3f}")


def main():
    print("=" * 96)
    print("Stage-1-frozen joint fit "
          "(η_para fitted on top of frozen Chinchilla + η_rep)")
    print("=" * 96)

    data = collect_pooled()
    n_per = {SOURCE_NONE:   int((data["source"] == SOURCE_NONE).sum()),
             SOURCE_REPEAT: int((data["source"] == SOURCE_REPEAT).sum()),
             SOURCE_PARA:   int((data["source"] == SOURCE_PARA).sum())}
    print(f"\nPooled: n_total={len(data['L'])}  "
          f"(1ep={n_per[SOURCE_NONE]}, rep={n_per[SOURCE_REPEAT]}, "
          f"para={n_per[SOURCE_PARA]})  "
          f"[14M excluded from 1ep+rep, kept for para]")

    print("\n[Stage 1] Fitting rep-only sub-model (9 params, no paraphrase)...")
    res1 = stage1_rep_only(data)
    p1 = res1["params"]
    print(f"  rep-only RMSE: {res1.get('rmse_logL', float('nan')):.4f}")
    _print_params("(stage 1 — rep-only baseline)", p1)

    fixed_keys = ["e", "a", "b", "alpha", "beta",
                  "log_K_rep", "rho_rep", "sigma_rep"]
    fixed_params: Dict[str, float] = {k: p1[k] for k in fixed_keys}

    print(f"\n[Stage 2] Fitting η_para only ({len(fixed_keys)} Stage-1 params "
          f"frozen)...")
    res2 = stage2_para_only(data, fixed_params)
    para_init = res2["params"]
    p2 = {**para_init, **fixed_params}
    _print_params("(stage 2 — η_para over frozen Stage 1)", p2)

    K_VALUES = [0, 5, 10, 15, 20, 25, 30]
    print(f"\n[Stage 3] Iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(data, para_init, fixed_params, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  | "
          f"{'lK_par':>7s} {'ρ_par':>7s} {'σ_par':>7s} | "
          f"{'1ep':>5s}  {'rep':>5s}  {'par':>5s}")
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(data, r["params"], r["pred_full"], r["keep"])
        p = r["params"]
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  | "
              f"{p['log_K_para']:>7.2f} {p['rho_para']:>+7.3f} "
              f"{p['sigma_para']:>+7.3f} | "
              f"{s['rmse_1ep']:>5.3f}  {s['rmse_rep']:>5.3f}  "
              f"{s['rmse_para']:>5.3f}")

    canonical = pick_canonical_k(sweep, data, K_VALUES, tol=0.001)
    print(f"\n  Canonical k (first para-RMSE plateau): k = {canonical}")
    rc = sweep[canonical]
    sc = _summary(data, rc["params"], rc["pred_full"], rc["keep"])
    _print_params(f"(canonical k={canonical})", rc["params"])
    print(f"    fit RMSE — 1ep: {sc['rmse_1ep']:.4f}  "
          f"rep: {sc['rmse_rep']:.4f}  para: {sc['rmse_para']:.4f}  "
          f"|  kept-total: {sc['rmse_kept']:.4f}")

    plot_diagnostic(
        data, rc["pred_full"], rc["keep"], rc["dropped"],
        rc["params"], sc,
        path=os.path.join(SCRIPT_DIR, "fit_joint_freeze.pdf"))

    return sweep, canonical


if __name__ == "__main__":
    main()
