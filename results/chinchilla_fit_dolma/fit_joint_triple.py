"""
Triple-joint one-shot fit on 1-epoch + multi-epoch *repetition* + *paraphrase*
data, all simultaneously.

Functional form (shared Chinchilla, separate η for repetition vs paraphrase):

    L(N, D, D'; src) = E + A/N^α + B / (D + η_src(D, D'; N) · D')^β

with src ∈ {none (1-epoch), repeat, para}.  η_src has the exp-sat
"Form B" shape with explicit R*(D, N):

    η_src(D, D'; N) = R*_src · (1 − e^{−x/R*_src}) / x,    x = D'/D,
    log R*_src = log K_src + ρ_src · log(D/N) + σ_src · log N.

For 1-epoch points D' = 0 → η has no effect.

We fit **11 parameters** simultaneously:
  Chinchilla   : e, a, b, α, β            (5 — *shared*)
  η_repeat     : log K_rep, ρ_rep, σ_rep  (3)
  η_paraphrase : log K_para, ρ_para, σ_para (3)

Pipeline:
  Stage 1: Fit the *9-param* sub-model (1-ep + multi-ep repetition only),
           Hoffmann-style grid search → reproduces fit_joint_all.py.
  Stage 2: Warm-start the full 11-param triple model from Stage 1,
           sweeping a grid only over (log K_para, ρ_para, σ_para).
  Stage 3: Iterative residual drop on the pooled (1-ep + rep + para)
           dataset, warm-starting each k.

Output:
  fit_joint_triple.pdf — three-panel diagnostic.
  Tabular comparison vs (a) two-stage 1-epoch-only fit,
                       (b) one-shot rep-only fit (writeup_final reference),
                       (c) triple fit (this script).
"""

import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))

# Palatino styling — match paper_figures.py
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

from data import (OVERFIT_EXCLUDE, SIZES, TTP_RATIO,  # noqa: E402
                  extract_1epoch, extract_multi_epoch,
                  extract_paraphrase, load, load_with_para)
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.1
SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA = 0, 1, 2

# Reference results we'll compare against (from writeup.md / writeup_final.md)
REF_1EP_ONLY = dict(  # Two-stage canonical k=20 (writeup §3.1)
    E=1.72, A=1115, B=20828, alpha=0.390, beta=0.451,
)
REF_ONESHOT_REP = dict(  # writeup_final headline (k=15)
    E=0.050, A=31.5, B=16539, alpha=0.137, beta=0.436,
    log_K_rep=10.32, rho_rep=-0.270, sigma_rep=-0.388,
)


# ──────────────────────────────────────────────────────────────────────
# Forward
# ──────────────────────────────────────────────────────────────────────

def _Rstar(log_K, rho, sigma, log_DoverN, log_N):
    return torch.exp(log_K + rho * log_DoverN + sigma * log_N)


def _eta_formB(R, x_safe):
    return R * (1.0 - torch.exp(-x_safe / R)) / x_safe


def make_triple_forward(N_arr, D_arr, Dp_arr, source_arr):
    """source_arr: 0=1ep, 1=repeat, 2=para. Returns forward(p) → log L."""
    log_N = torch.tensor(np.log(N_arr), dtype=torch.float64)
    D_t  = torch.tensor(D_arr, dtype=torch.float64)
    Dp_t = torch.tensor(Dp_arr, dtype=torch.float64)
    src_t = torch.tensor(source_arr, dtype=torch.int64)
    log_DoverN = torch.log(D_t / torch.tensor(N_arr, dtype=torch.float64))
    is_rep  = (src_t == SOURCE_REPEAT)
    is_para = (src_t == SOURCE_PARA)
    has_Dp  = is_rep | is_para
    # Safe x = D'/D (=1 where Dp=0, gets multiplied by 0 anyway)
    safe_D = torch.where(D_t > 0, D_t, torch.ones_like(D_t))
    x = Dp_t / safe_D
    safe_x = torch.where(has_Dp, x, torch.ones_like(x))

    def forward(p):
        # η_rep
        R_rep = _Rstar(p["log_K_rep"], p["rho_rep"], p["sigma_rep"],
                        log_DoverN, log_N)
        eta_rep = _eta_formB(R_rep, safe_x)
        # η_para
        R_pa = _Rstar(p["log_K_para"], p["rho_para"], p["sigma_para"],
                       log_DoverN, log_N)
        eta_pa = _eta_formB(R_pa, safe_x)
        # Pick the right η per row, 0 for 1-epoch
        eta = torch.where(is_rep, eta_rep,
                           torch.where(is_para, eta_pa,
                                        torch.zeros_like(eta_rep)))
        D_eff = D_t + eta * Dp_t  # = D where Dp = 0
        log_D_eff = torch.log(D_eff)
        terms = torch.stack([
            p["e"].expand_as(log_N),
            p["a"] - p["alpha"] * log_N,
            p["b"] - p["beta"] * log_D_eff,
        ], dim=0)
        return logsumexp_stable(terms, dim=0)

    return forward


def make_forward_rep_only(N_arr, D_arr, Dp_arr, is_multi_arr):
    """Stage-1 sub-model: 9 params (Chinchilla + η_rep). Used as warm-start
    seed for the full triple fit, and as the writeup-final reference fit."""
    log_N = torch.tensor(np.log(N_arr), dtype=torch.float64)
    D_t  = torch.tensor(D_arr, dtype=torch.float64)
    Dp_t = torch.tensor(Dp_arr, dtype=torch.float64)
    has_Dp = torch.tensor(is_multi_arr, dtype=torch.bool)
    log_DoverN = torch.log(D_t / torch.tensor(N_arr, dtype=torch.float64))
    safe_D = torch.where(D_t > 0, D_t, torch.ones_like(D_t))
    x = Dp_t / safe_D
    safe_x = torch.where(has_Dp, x, torch.ones_like(x))

    def forward(p):
        R = _Rstar(p["log_K_rep"], p["rho_rep"], p["sigma_rep"],
                    log_DoverN, log_N)
        eta = _eta_formB(R, safe_x)
        eta = torch.where(has_Dp, eta, torch.zeros_like(eta))
        D_eff = D_t + eta * Dp_t
        log_D_eff = torch.log(D_eff)
        terms = torch.stack([
            p["e"].expand_as(log_N),
            p["a"] - p["alpha"] * log_N,
            p["b"] - p["beta"] * log_D_eff,
        ], dim=0)
        return logsumexp_stable(terms, dim=0)

    return forward


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def collect_pooled_triple(scale_min_para: float = 0.5):
    """1-ep + multi-ep (repetition) + paraphrase pooled across sizes.

    `scale_min_para` filters the paraphrase corpus by chinchilla scale
    (D = chinchilla_scale * 20 * N). Default 0.5 reproduces the writeup
    §5/§6 corpus (63 paraphrase points). Set to 0.0 to include the
    small-scale paraphrase rows added after §6 (used in §7's quad fit).

    Returns dict with arrays N, D, Dp, L, source (0/1/2), tag.
    """
    tags, Ns, Ds, Dps, Ls, src = [], [], [], [], [], []
    for size in SIZES:
        N, datasets, parap = load_with_para(size)
        # 1-epoch
        s1, D1, L1 = extract_1epoch(datasets, N, scale_min=0.0)
        for d, l in zip(D1, L1):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(0.0)
            Ls.append(l); src.append(SOURCE_NONE)
        # repetition
        sm, Dm, em, Dpm, Lm, _L1m = extract_multi_epoch(
            datasets, N, scale_min=0.0,
            exclude_overfit=OVERFIT_EXCLUDE.get(size, set()))
        for d, dp, l in zip(Dm, Dpm, Lm):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
            Ls.append(l); src.append(SOURCE_REPEAT)
        # paraphrase
        if parap:
            sp, Dp_, Kp_, Dpp, Lp, _ = extract_paraphrase(
                datasets, parap, N, scale_min=scale_min_para)
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
# Fitting machinery
# ──────────────────────────────────────────────────────────────────────

# Hoffmann-style grid for the 5 chinchilla params + 3 η_rep params
GRID_REP_ONLY = expand_grid({
    "e":           [0.0, 1.0],
    "a":           [3.0, 12.0],
    "b":           [5.0, 12.0],
    "alpha":       [0.15, 0.4],
    "beta":        [0.3, 0.5],
    "log_K_rep":   [10.0, 18.0],
    "rho_rep":     [-1.0, 0.0],
    "sigma_rep":   [-1.0, 0.0],
})  # 2^8 = 256 inits

# Para η-grid for stage 2 warm-start
PARA_GRID = [
    {"log_K_para": lk, "rho_para": rp, "sigma_para": sg}
    for lk in [10.0, 14.0, 18.0]
    for rp in [-1.0, 0.0, 1.0, 2.0]
    for sg in [-1.0, 0.0]
]  # 24 inits


def stage1_rep_only(data, delta=DELTA):
    """Fit 9-param (Chinchilla + η_rep) on 1-ep + multi-ep, no paraphrase."""
    keep = data["source"] != SOURCE_PARA
    fwd = make_forward_rep_only(
        data["N"][keep], data["D"][keep], data["Dp"][keep],
        is_multi_arr=(data["source"][keep] == SOURCE_REPEAT))
    log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
    return fit_lse(fwd, log_L, GRID_REP_ONLY, delta=delta, verbose=False)


def stage2_triple(data, warm_rep_params, delta=DELTA):
    """Fit 11-param triple model, warm-starting from stage 1 + para grid."""
    init_grid = [{**{k: float(v) for k, v in warm_rep_params.items()}, **g}
                 for g in PARA_GRID]
    fwd = make_triple_forward(data["N"], data["D"], data["Dp"], data["source"])
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    return fit_lse(fwd, log_L, init_grid, delta=delta, verbose=False)


def topk_drop_sweep(data, init_params, k_values, delta=DELTA):
    """Iterative residual-greedy drop on pooled data, warm-started from
    init_params at every k."""
    fwd_full = make_triple_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    last = init_params
    keep = np.ones(len(data["L"]), dtype=bool)
    cumulative = []
    out = {}
    for k in sorted(set(k_values)):
        while len(cumulative) < k:
            grid = [{kk: float(vv) for kk, vv in last.items()}]
            fwd = make_triple_forward(
                data["N"][keep], data["D"][keep],
                data["Dp"][keep], data["source"][keep])
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
        fwd = make_triple_forward(
            data["N"][keep], data["D"][keep],
            data["Dp"][keep], data["source"][keep])
        log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
        res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
        last = res["params"]
        with torch.no_grad():
            p_t = {kk: torch.tensor(v, dtype=torch.float64)
                   for kk, v in res["params"].items()}
            pred_full = fwd_full(p_t).numpy()
        out[k] = dict(params=res["params"],
                       pred_full=pred_full,
                       keep=keep.copy(),
                       dropped=list(cumulative))
    return out


def _summary(data, params, pred_full, keep):
    log_L = np.log(data["L"])
    resid = log_L - pred_full
    summary = dict(rmse_full=float(np.sqrt(np.mean(resid ** 2))),
                   rmse_kept=float(np.sqrt(np.mean(resid[keep] ** 2))))
    for tag, src_id in [("1ep", SOURCE_NONE), ("rep", SOURCE_REPEAT),
                        ("para", SOURCE_PARA)]:
        m = (data["source"] == src_id)
        mk = m & keep
        summary[f"n_{tag}"] = int(m.sum())
        summary[f"n_{tag}_kept"] = int(mk.sum())
        summary[f"rmse_{tag}"] = (float(np.sqrt(np.mean(resid[mk] ** 2)))
                                   if mk.any() else float("nan"))
    summary["params"] = params
    return summary


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    return f"{x:.0f}"


def plot_triple_diagnostic(data, pred_full, keep, dropped,
                            params, summary, path):
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
        # 1-ep — circles, plotted vs D
        m1 = m & (data["source"] == SOURCE_NONE)
        ax_data.scatter(data["D"][m1], data["L"][m1], s=70, color=c,
                        edgecolors="k", linewidths=0.4, marker="o",
                        zorder=4, label=tag)
        # rep — × markers, plotted vs D + Dp (total tokens)
        mr = m & (data["source"] == SOURCE_REPEAT)
        ax_data.scatter(data["D"][mr] + data["Dp"][mr], data["L"][mr],
                        s=42, color=c, edgecolors="k", linewidths=0.2,
                        marker="x", alpha=0.7, zorder=3)
        # para — squares, plotted vs D + Dp
        mp = m & (data["source"] == SOURCE_PARA)
        ax_data.scatter(data["D"][mp] + data["Dp"][mp], data["L"][mp],
                        s=42, color=c, edgecolors="k", linewidths=0.5,
                        marker="s", alpha=0.85, zorder=3)
        # residuals coloured by source via marker
        for src_id, mk_marker in [(SOURCE_NONE, "o"),
                                   (SOURCE_REPEAT, "x"),
                                   (SOURCE_PARA, "s")]:
            mm = m & (data["source"] == src_id)
            if mm.any():
                ax_res.scatter(data["D"][mm], resid[mm], s=40, color=c,
                               edgecolors="k" if mk_marker != "x" else "none",
                               linewidths=0.3, marker=mk_marker)
        # parity
        ax_par.scatter(np.exp(pred_full[m]), data["L"][m], s=42, color=c,
                       edgecolors="k", linewidths=0.3)
    # Mark dropped points
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
        rf"Triple fit: $E={np.exp(p['e']):.2f}$, $A={np.exp(p['a']):.0f}$, "
        rf"$B={np.exp(p['b']):.0f}$, $\alpha={p['alpha']:.2f}$, "
        rf"$\beta={p['beta']:.3f}$" + "\n"
        rf"η_rep: $\log K={p['log_K_rep']:.1f}$, $\rho={p['rho_rep']:+.2f}$, "
        rf"$\sigma={p['sigma_rep']:+.2f}$  |  "
        rf"η_para: $\log K={p['log_K_para']:.1f}$, $\rho={p['rho_para']:+.2f}$, "
        rf"$\sigma={p['sigma_para']:+.2f}$",
        fontsize=11)

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_res.set_xlabel(r"$D$")
    ax_res.set_ylabel(r"residual ($\log L$)")
    ax_res.set_title(rf"Residuals  (kept RMSE: 1ep={summary['rmse_1ep']:.3f}, "
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
        "Triple-joint fit on 1-epoch + repetition + paraphrase  "
        "(○ 1ep, × repeat, □ para)",
        fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _print_params(label, p):
    items = [(k, p[k]) for k in
             ["e", "a", "b", "alpha", "beta",
              "log_K_rep", "rho_rep", "sigma_rep",
              "log_K_para", "rho_para", "sigma_para"]
             if k in p]
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
    print("Triple-joint fit: 1-epoch + repetition + paraphrase  "
          "(shared E, A, B, α, β; separate η_rep, η_para)")
    print("=" * 96)

    data = collect_pooled_triple()
    n_per = {SOURCE_NONE: int((data["source"] == SOURCE_NONE).sum()),
             SOURCE_REPEAT: int((data["source"] == SOURCE_REPEAT).sum()),
             SOURCE_PARA: int((data["source"] == SOURCE_PARA).sum())}
    print(f"\nPooled: n_total={len(data['L'])}  "
          f"(1ep={n_per[SOURCE_NONE]}, "
          f"rep={n_per[SOURCE_REPEAT]}, para={n_per[SOURCE_PARA]})")

    # ── Stage 1: 9-param rep-only fit (writeup_final reference) ────
    print("\n[Stage 1] Fitting rep-only sub-model (9 params, no paraphrase)...")
    res1 = stage1_rep_only(data)
    p1 = res1["params"]
    print(f"  rep-only RMSE: {res1.get('rmse_logL', float('nan')):.4f}")
    _print_params("(stage 1 — rep-only baseline)", p1)

    # ── Stage 2: 11-param triple, warm-start from stage 1 ─────────
    print("\n[Stage 2] Stage-2 triple-model fit (11 params, warm-started "
          f"with {len(PARA_GRID)} para-grid inits)...")
    res2 = stage2_triple(data, p1)
    p2 = res2["params"]
    _print_params("(stage 2 — full triple before residual drop)", p2)

    # ── Stage 3: residual drop sweep ───────────────────────────────
    K_VALUES = [0, 5, 10, 15, 20, 25, 30]
    print(f"\n[Stage 3] Iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(data, p2, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  {'E':>7s}  {'A':>7s}  "
          f"{'B':>9s}  {'α':>6s}  {'β':>6s}  | "
          f"{'lK_rep':>7s} {'ρ_rep':>7s} {'σ_rep':>7s} | "
          f"{'lK_par':>7s} {'ρ_par':>7s} {'σ_par':>7s} | "
          f"{'1ep':>5s}  {'rep':>5s}  {'par':>5s}")
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(data, r["params"], r["pred_full"], r["keep"])
        p = r["params"]
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  "
              f"{np.exp(p['e']):>7.3f}  {np.exp(p['a']):>7.1f}  "
              f"{np.exp(p['b']):>9.0f}  {p['alpha']:>6.3f}  "
              f"{p['beta']:>6.3f}  | "
              f"{p['log_K_rep']:>7.2f} {p['rho_rep']:>+7.3f} "
              f"{p['sigma_rep']:>+7.3f} | "
              f"{p['log_K_para']:>7.2f} {p['rho_para']:>+7.3f} "
              f"{p['sigma_para']:>+7.3f} | "
              f"{s['rmse_1ep']:>5.3f}  {s['rmse_rep']:>5.3f}  "
              f"{s['rmse_para']:>5.3f}")

    # Pick canonical k (first |Δβ|<0.01 break)
    betas = [sweep[k]["params"]["beta"] for k in K_VALUES]
    canonical = K_VALUES[-1]
    for i in range(1, len(betas)):
        if abs(betas[i] - betas[i - 1]) < 0.01:
            canonical = K_VALUES[i]; break
    print(f"\n  Canonical k (|Δβ|<0.01 first break): k = {canonical}")
    rc = sweep[canonical]
    sc = _summary(data, rc["params"], rc["pred_full"], rc["keep"])
    _print_params(f"(stage 3 canonical k={canonical})", rc["params"])
    print(f"    fit RMSE — 1ep: {sc['rmse_1ep']:.4f}  "
          f"rep: {sc['rmse_rep']:.4f}  para: {sc['rmse_para']:.4f}  "
          f"|  kept-total: {sc['rmse_kept']:.4f}")

    # ── Side-by-side comparison vs reference fits ──────────────────
    print("\n" + "=" * 96)
    print("Comparison with previously-reported fits:")
    print("=" * 96)
    print(f"  {'fit':<42s}  {'E':>6s}  {'A':>6s}  {'B':>7s}  "
          f"{'α':>5s}  {'β':>5s}  {'ηrep_logK':>9s}  {'ηpara_logK':>10s}")
    print(f"  {'-'*42:<42s}  {'-'*6:>6s}  {'-'*6:>6s}  {'-'*7:>7s}  "
          f"{'-'*5:>5s}  {'-'*5:>5s}  {'-'*9:>9s}  {'-'*10:>10s}")
    print(f"  {'two-stage 1-ep only (k=20, writeup §3.1)':<42s}  "
          f"{REF_1EP_ONLY['E']:>6.2f}  {REF_1EP_ONLY['A']:>6.0f}  "
          f"{REF_1EP_ONLY['B']:>7.0f}  {REF_1EP_ONLY['alpha']:>5.3f}  "
          f"{REF_1EP_ONLY['beta']:>5.3f}  {'-':>9s}  {'-':>10s}")
    print(f"  {'one-shot rep+1ep (k=15, writeup_final)':<42s}  "
          f"{REF_ONESHOT_REP['E']:>6.2f}  {REF_ONESHOT_REP['A']:>6.1f}  "
          f"{REF_ONESHOT_REP['B']:>7.0f}  {REF_ONESHOT_REP['alpha']:>5.3f}  "
          f"{REF_ONESHOT_REP['beta']:>5.3f}  "
          f"{REF_ONESHOT_REP['log_K_rep']:>9.2f}  {'-':>10s}")
    pc = rc["params"]
    print(f"  {f'TRIPLE (k={canonical}, this script)':<42s}  "
          f"{np.exp(pc['e']):>6.2f}  {np.exp(pc['a']):>6.1f}  "
          f"{np.exp(pc['b']):>7.0f}  {pc['alpha']:>5.3f}  "
          f"{pc['beta']:>5.3f}  {pc['log_K_rep']:>9.2f}  "
          f"{pc['log_K_para']:>10.2f}")

    plot_triple_diagnostic(
        data, rc["pred_full"], rc["keep"], rc["dropped"],
        rc["params"], sc,
        path=os.path.join(SCRIPT_DIR, "fit_joint_triple.pdf"))

    return sweep, canonical


if __name__ == "__main__":
    main()
