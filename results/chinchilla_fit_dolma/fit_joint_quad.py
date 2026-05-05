"""
Quad-joint one-shot fit on 1-epoch + multi-epoch *repetition* + *paraphrase*
+ *self-distill* data, all simultaneously.

Functional form (shared Chinchilla, separate η for each "second stream"):

    L(N, D, D'; src) = E + A/N^α + B / (D + η_src(D, D'; N) · D')^β

with src ∈ {none (1-epoch), repeat, para, sd}.  η_src is exp-sat
"Form B" with explicit R*(D, N):

    η_src(D, D'; N) = R*_src · (1 − e^{−x/R*_src}) / x,    x = D'/D,
    log R*_src = log K_src + ρ_src · log(D/N) + σ_src · log N.

We fit **14 parameters**:
  Chinchilla   : e, a, b, α, β            (5 — *shared*)
  η_repeat     : log K_rep, ρ_rep, σ_rep  (3)
  η_paraphrase : log K_para, ρ_para, σ_para (3)
  η_selfdistill: log K_sd, ρ_sd, σ_sd     (3 — σ_sd is unidentified
                                          since SD is 30M-only; reported
                                          for completeness)

Pipeline mirrors fit_joint_triple.py:
  Stage 1: 9-param rep+1ep fit (writeup_final reference reproduction).
  Stage 2: 11-param triple by warm-starting + para grid.
  Stage 3: 14-param quad by warm-starting + SD grid.
  Stage 4: iterative residual drop on the pooled (1ep + rep + para + SD).
"""

import glob
import os
import sys
from typing import Dict, List, Optional

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

from data import (OVERFIT_EXCLUDE, SIZES, TTP_RATIO,  # noqa: E402
                  extract_1epoch, extract_multi_epoch,
                  extract_paraphrase, extract_selfdistill,
                  load_with_extras)
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.1
SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA, SOURCE_SD = 0, 1, 2, 3

REF_1EP_ONLY = dict(E=1.72, A=1115, B=20828, alpha=0.390, beta=0.451)
REF_ONESHOT_REP = dict(E=0.050, A=31.5, B=16539, alpha=0.137, beta=0.436,
                        log_K_rep=10.32, rho_rep=-0.270, sigma_rep=-0.388)
# Triple fit (k=15) from fit_joint_triple.py
REF_TRIPLE_K15 = dict(E=0.003, A=28.9, B=15599, alpha=0.133, beta=0.431,
                       log_K_rep=10.58, rho_rep=-0.414, sigma_rep=-0.394,
                       log_K_para=10.10, rho_para=-2.555, sigma_para=+0.177)


# ──────────────────────────────────────────────────────────────────────
# Forward
# ──────────────────────────────────────────────────────────────────────

def _Rstar(log_K, rho, sigma, log_DoverN, log_N):
    return torch.exp(log_K + rho * log_DoverN + sigma * log_N)


def _eta_formB(R, x_safe):
    return R * (1.0 - torch.exp(-x_safe / R)) / x_safe


def make_quad_forward(N_arr, D_arr, Dp_arr, source_arr):
    log_N = torch.tensor(np.log(N_arr), dtype=torch.float64)
    D_t  = torch.tensor(D_arr, dtype=torch.float64)
    Dp_t = torch.tensor(Dp_arr, dtype=torch.float64)
    src_t = torch.tensor(source_arr, dtype=torch.int64)
    log_DoverN = torch.log(D_t / torch.tensor(N_arr, dtype=torch.float64))
    is_rep  = (src_t == SOURCE_REPEAT)
    is_para = (src_t == SOURCE_PARA)
    is_sd   = (src_t == SOURCE_SD)
    has_Dp  = is_rep | is_para | is_sd
    safe_D = torch.where(D_t > 0, D_t, torch.ones_like(D_t))
    x = Dp_t / safe_D
    safe_x = torch.where(has_Dp, x, torch.ones_like(x))

    def forward(p):
        R_rep  = _Rstar(p["log_K_rep"],  p["rho_rep"],  p["sigma_rep"],
                         log_DoverN, log_N)
        R_para = _Rstar(p["log_K_para"], p["rho_para"], p["sigma_para"],
                         log_DoverN, log_N)
        R_sd   = _Rstar(p["log_K_sd"],   p["rho_sd"],   p["sigma_sd"],
                         log_DoverN, log_N)
        eta_rep  = _eta_formB(R_rep,  safe_x)
        eta_para = _eta_formB(R_para, safe_x)
        eta_sd   = _eta_formB(R_sd,   safe_x)
        eta = torch.where(is_rep,  eta_rep,
                torch.where(is_para, eta_para,
                  torch.where(is_sd, eta_sd, torch.zeros_like(eta_rep))))
        D_eff = D_t + eta * Dp_t
        log_D_eff = torch.log(D_eff)
        terms = torch.stack([
            p["e"].expand_as(log_N),
            p["a"] - p["alpha"] * log_N,
            p["b"] - p["beta"] * log_D_eff,
        ], dim=0)
        return logsumexp_stable(terms, dim=0)

    return forward


def make_triple_forward(N_arr, D_arr, Dp_arr, source_arr):
    """3-stream version (no SD), used in stage-2 warm-start."""
    log_N = torch.tensor(np.log(N_arr), dtype=torch.float64)
    D_t  = torch.tensor(D_arr, dtype=torch.float64)
    Dp_t = torch.tensor(Dp_arr, dtype=torch.float64)
    src_t = torch.tensor(source_arr, dtype=torch.int64)
    log_DoverN = torch.log(D_t / torch.tensor(N_arr, dtype=torch.float64))
    is_rep  = (src_t == SOURCE_REPEAT)
    is_para = (src_t == SOURCE_PARA)
    has_Dp  = is_rep | is_para
    safe_D = torch.where(D_t > 0, D_t, torch.ones_like(D_t))
    x = Dp_t / safe_D
    safe_x = torch.where(has_Dp, x, torch.ones_like(x))

    def forward(p):
        R_rep  = _Rstar(p["log_K_rep"],  p["rho_rep"],  p["sigma_rep"],
                         log_DoverN, log_N)
        R_para = _Rstar(p["log_K_para"], p["rho_para"], p["sigma_para"],
                         log_DoverN, log_N)
        eta_rep  = _eta_formB(R_rep,  safe_x)
        eta_para = _eta_formB(R_para, safe_x)
        eta = torch.where(is_rep, eta_rep,
                           torch.where(is_para, eta_para,
                                        torch.zeros_like(eta_rep)))
        D_eff = D_t + eta * Dp_t
        log_D_eff = torch.log(D_eff)
        terms = torch.stack([
            p["e"].expand_as(log_N),
            p["a"] - p["alpha"] * log_N,
            p["b"] - p["beta"] * log_D_eff,
        ], dim=0)
        return logsumexp_stable(terms, dim=0)

    return forward


def make_rep_only_forward(N_arr, D_arr, Dp_arr, is_multi_arr):
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

def collect_pooled_quad():
    tags, Ns, Ds, Dps, Ls, src = [], [], [], [], [], []
    for size in SIZES:
        N, datasets, parap, sd_ds = load_with_extras(size)
        s1, D1, L1 = extract_1epoch(datasets, N, scale_min=0.0)
        for d, l in zip(D1, L1):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(0.0)
            Ls.append(l); src.append(SOURCE_NONE)
        sm, Dm, em, Dpm, Lm, _ = extract_multi_epoch(
            datasets, N, scale_min=0.0,
            exclude_overfit=OVERFIT_EXCLUDE.get(size, set()))
        for d, dp, l in zip(Dm, Dpm, Lm):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
            Ls.append(l); src.append(SOURCE_REPEAT)
        if parap:
            sp, Dp_, _, Dpp, Lp, _ = extract_paraphrase(
                datasets, parap, N, scale_min=0.0)
            for d, dp, l in zip(Dp_, Dpp, Lp):
                tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
                Ls.append(l); src.append(SOURCE_PARA)
        if sd_ds:
            ss, Ds_sd, _, Dp_sd, Lsd, _ = extract_selfdistill(
                datasets, sd_ds, N, scale_min=0.0)
            for d, dp, l in zip(Ds_sd, Dp_sd, Lsd):
                tags.append(size); Ns.append(N); Ds.append(d); Dps.append(dp)
                Ls.append(l); src.append(SOURCE_SD)
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

GRID_REP_ONLY = expand_grid({
    "e":           [0.0, 1.0],
    "a":           [3.0, 12.0],
    "b":           [5.0, 12.0],
    "alpha":       [0.15, 0.4],
    "beta":        [0.3, 0.5],
    "log_K_rep":   [10.0, 18.0],
    "rho_rep":     [-1.0, 0.0],
    "sigma_rep":   [-1.0, 0.0],
})

PARA_GRID = [
    {"log_K_para": lk, "rho_para": rp, "sigma_para": sg}
    for lk in [10.0, 14.0, 18.0]
    for rp in [-2.0, -0.5, 1.0, 2.0]
    for sg in [-1.0, 0.0, 1.0]
]

SD_GRID = [
    {"log_K_sd": lk, "rho_sd": rp, "sigma_sd": sg}
    for lk in [8.0, 14.0, 18.0]
    for rp in [-1.0, 0.0, 1.0]
    for sg in [-1.0, 0.0]   # note: σ_sd is unidentified (only one N for SD)
]


def stage1_rep_only(data, delta=DELTA):
    keep = (data["source"] == SOURCE_NONE) | (data["source"] == SOURCE_REPEAT)
    fwd = make_rep_only_forward(
        data["N"][keep], data["D"][keep], data["Dp"][keep],
        is_multi_arr=(data["source"][keep] == SOURCE_REPEAT))
    log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
    return fit_lse(fwd, log_L, GRID_REP_ONLY, delta=delta, verbose=False)


def stage2_triple(data, warm_rep_params, delta=DELTA):
    keep = data["source"] != SOURCE_SD
    init_grid = [{**{k: float(v) for k, v in warm_rep_params.items()}, **g}
                 for g in PARA_GRID]
    fwd = make_triple_forward(
        data["N"][keep], data["D"][keep],
        data["Dp"][keep], data["source"][keep])
    log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
    return fit_lse(fwd, log_L, init_grid, delta=delta, verbose=False)


def stage3_quad(data, warm_triple_params, delta=DELTA):
    init_grid = [{**{k: float(v) for k, v in warm_triple_params.items()}, **g}
                 for g in SD_GRID]
    fwd = make_quad_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    return fit_lse(fwd, log_L, init_grid, delta=delta, verbose=False)


def topk_drop_sweep(data, init_params, k_values, delta=DELTA):
    fwd_full = make_quad_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    last = init_params
    keep = np.ones(len(data["L"]), dtype=bool)
    cumulative = []
    out = {}
    for k in sorted(set(k_values)):
        while len(cumulative) < k:
            grid = [{kk: float(vv) for kk, vv in last.items()}]
            fwd = make_quad_forward(
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
        grid = [{kk: float(vv) for kk, vv in last.items()}]
        fwd = make_quad_forward(
            data["N"][keep], data["D"][keep],
            data["Dp"][keep], data["source"][keep])
        log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
        res = fit_lse(fwd, log_L, grid, delta=delta, verbose=False)
        last = res["params"]
        with torch.no_grad():
            p_t = {kk: torch.tensor(v, dtype=torch.float64)
                   for kk, v in res["params"].items()}
            pred_full = fwd_full(p_t).numpy()
        out[k] = dict(params=res["params"], pred_full=pred_full,
                       keep=keep.copy(), dropped=list(cumulative))
    return out


def _summary(data, params, pred_full, keep):
    log_L = np.log(data["L"]); resid = log_L - pred_full
    summary = dict(rmse_full=float(np.sqrt(np.mean(resid ** 2))),
                   rmse_kept=float(np.sqrt(np.mean(resid[keep] ** 2))))
    for tag, src_id in [("1ep", SOURCE_NONE), ("rep", SOURCE_REPEAT),
                        ("para", SOURCE_PARA), ("sd", SOURCE_SD)]:
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


def plot_quad_diagnostic(data, pred_full, keep, dropped, params, summary,
                          path):
    sizes = sorted(set(data["tags"].tolist()), key=lambda t: SIZES[t][0])
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sizes)))
    color_of = {s: cmap[i] for i, s in enumerate(sizes)}

    log_L = np.log(data["L"])
    resid = log_L - pred_full
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_data, ax_res, ax_par = axes

    SRC_MARKER = {SOURCE_NONE: "o", SOURCE_REPEAT: "x",
                  SOURCE_PARA: "s", SOURCE_SD: "D"}
    for tag in sizes:
        m = data["tags"] == tag
        c = color_of[tag]
        for src_id in [SOURCE_NONE, SOURCE_REPEAT, SOURCE_PARA, SOURCE_SD]:
            mm = m & (data["source"] == src_id)
            if not mm.any(): continue
            mk = SRC_MARKER[src_id]
            x = data["D"][mm] + (data["Dp"][mm] if src_id != SOURCE_NONE else 0)
            edge = "k" if mk != "x" else "none"
            ax_data.scatter(x, data["L"][mm], s=55, color=c,
                            edgecolors=edge, linewidths=0.3,
                            marker=mk, alpha=0.85,
                            label=tag if src_id == SOURCE_NONE else None)
            ax_res.scatter(data["D"][mm], resid[mm], s=40, color=c,
                           edgecolors=edge, linewidths=0.25, marker=mk)
        ax_par.scatter(np.exp(pred_full[m]), data["L"][m], s=40, color=c,
                       edgecolors="k", linewidths=0.25)
    if len(dropped):
        dmask = np.zeros(len(data["L"]), dtype=bool); dmask[dropped] = True
        ax_data.scatter(
            data["D"][dmask] + data["Dp"][dmask], data["L"][dmask],
            s=180, facecolors="none", edgecolors="red",
            linewidths=1.6, zorder=10, label="dropped")

    ax_data.set_xscale("log", base=2)
    ax_data.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_data.set_xlabel(r"Effective tokens $D + D'$")
    ax_data.set_ylabel("Validation loss")
    ax_data.legend(loc="upper right", ncol=2, fontsize=9)
    ax_data.grid(alpha=0.3)
    p = params
    ax_data.set_title(
        rf"Quad fit: $E={np.exp(p['e']):.2f}$, $A={np.exp(p['a']):.0f}$, "
        rf"$B={np.exp(p['b']):.0f}$, $\alpha={p['alpha']:.2f}$, "
        rf"$\beta={p['beta']:.3f}$" + "\n"
        rf"η_rep ($\log K{{=}}{p['log_K_rep']:.1f}$, "
        rf"$\rho{{=}}{p['rho_rep']:+.2f}$, $\sigma{{=}}{p['sigma_rep']:+.2f}$)  |  "
        rf"η_para ($\log K{{=}}{p['log_K_para']:.1f}$, "
        rf"$\rho{{=}}{p['rho_para']:+.2f}$, $\sigma{{=}}{p['sigma_para']:+.2f}$)  |  "
        rf"η_sd ($\log K{{=}}{p['log_K_sd']:.1f}$, "
        rf"$\rho{{=}}{p['rho_sd']:+.2f}$, $\sigma{{=}}{p['sigma_sd']:+.2f}$)",
        fontsize=10)

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_res.set_xlabel(r"$D$")
    ax_res.set_ylabel(r"residual ($\log L$)")
    ax_res.set_title(rf"Residuals  (kept RMSE: 1ep={summary['rmse_1ep']:.3f}, "
                     rf"rep={summary['rmse_rep']:.3f}, "
                     rf"para={summary['rmse_para']:.3f}, "
                     rf"sd={summary['rmse_sd']:.3f})")
    ax_res.grid(alpha=0.3)

    lo = min(np.exp(pred_full).min(), data["L"].min()) * 0.95
    hi = max(np.exp(pred_full).max(), data["L"].max()) * 1.05
    ax_par.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6)
    ax_par.set_xlabel("Predicted L")
    ax_par.set_ylabel("Observed L")
    ax_par.set_title("Parity")
    ax_par.grid(alpha=0.3)

    fig.suptitle(
        "Quad-joint fit on 1-epoch + repetition + paraphrase + self-distill"
        "  (○ 1ep, × repeat, □ para, ◇ sd)",
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
    if "log_K_sd" in p:
        print(f"    η_sd:       log K={p['log_K_sd']:.2f}  "
              f"ρ={p['rho_sd']:+.3f}  σ={p['sigma_sd']:+.3f}  "
              f"(σ_sd is unidentified — SD only at one N)")


def main():
    print("=" * 96)
    print("Quad-joint fit: 1-epoch + repetition + paraphrase + self-distill")
    print("=" * 96)

    data = collect_pooled_quad()
    n_per = {SOURCE_NONE: int((data["source"] == SOURCE_NONE).sum()),
             SOURCE_REPEAT: int((data["source"] == SOURCE_REPEAT).sum()),
             SOURCE_PARA: int((data["source"] == SOURCE_PARA).sum()),
             SOURCE_SD: int((data["source"] == SOURCE_SD).sum())}
    print(f"\nPooled: n_total={len(data['L'])}  "
          f"(1ep={n_per[SOURCE_NONE]}, rep={n_per[SOURCE_REPEAT]}, "
          f"para={n_per[SOURCE_PARA]}, sd={n_per[SOURCE_SD]})")

    print("\n[Stage 1] rep-only sub-model (9 params)...")
    res1 = stage1_rep_only(data); p1 = res1["params"]
    _print_params("(stage 1 baseline)", p1)

    print("\n[Stage 2] triple model (warm-start + para grid)...")
    res2 = stage2_triple(data, p1); p2 = res2["params"]
    _print_params("(stage 2 — triple from rep-only seed)", p2)

    # Stage 2 occasionally falls into a low-β basin when the para grid
    # opens up new attractors with the small-scale paraphrase data added.
    # Anchor a *second* initialization at the published TRIPLE k=15 params
    # (writeup §6) and re-fit; pick whichever has lower in-sample loss.
    print("\n[Stage 2'] alternate warm-start from TRIPLE k=15 params...")
    rt = REF_TRIPLE_K15
    triple_anchor = dict(
        e=float(np.log(rt["E"])), a=float(np.log(rt["A"])),
        b=float(np.log(rt["B"])), alpha=rt["alpha"], beta=rt["beta"],
        log_K_rep=rt["log_K_rep"], rho_rep=rt["rho_rep"],
        sigma_rep=rt["sigma_rep"],
        log_K_para=rt["log_K_para"], rho_para=rt["rho_para"],
        sigma_para=rt["sigma_para"],
    )
    fwd_triple = make_triple_forward(
        data["N"][data["source"] != SOURCE_SD],
        data["D"][data["source"] != SOURCE_SD],
        data["Dp"][data["source"] != SOURCE_SD],
        data["source"][data["source"] != SOURCE_SD])
    log_L_t = torch.tensor(
        np.log(data["L"][data["source"] != SOURCE_SD]),
        dtype=torch.float64)
    res2b = fit_lse(fwd_triple, log_L_t,
                     [{k: float(v) for k, v in triple_anchor.items()}],
                     delta=DELTA, verbose=False)
    p2b = res2b["params"]
    _print_params("(stage 2' — triple from TRIPLE-k=15 seed)", p2b)
    # Pick the better stage-2 result by in-sample Huber-loss
    p2_best = p2b if res2b.get("rmse_logL", float("inf")) < \
                     res2.get("rmse_logL", float("inf")) else p2
    print(f"  → using {'TRIPLE-k=15 seed' if p2_best is p2b else 'rep-only seed'} "
          f"for stage 3 (RMSE: rep-seed={res2.get('rmse_logL', float('nan')):.4f}, "
          f"triple-seed={res2b.get('rmse_logL', float('nan')):.4f})")

    print("\n[Stage 3] quad model (warm-start + SD grid)...")
    res3 = stage3_quad(data, p2_best); p3 = res3["params"]
    _print_params("(stage 3 — quad before residual drop)", p3)

    # Belt-and-braces: also try the published TRIPLE-k=15 params + sensible
    # SD init as a *direct* starting point for the quad — bypass stage 1+2
    # entirely, which can fall into low-β local minima.
    print("\n[Stage 3'] direct fit from TRIPLE-k=15 + SD seed grid...")
    triple_anchor_full = {**triple_anchor,
                           "log_K_sd": 10.0, "rho_sd": -1.0, "sigma_sd": -0.5}
    sd_alt_grid = [{**triple_anchor_full, "log_K_sd": lk,
                     "rho_sd": rp, "sigma_sd": sg}
                    for lk in [8.0, 10.0, 14.0]
                    for rp in [-2.0, -1.0, 0.0]
                    for sg in [-1.0, 0.0]]
    fwd_quad = make_quad_forward(
        data["N"], data["D"], data["Dp"], data["source"])
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    res3b = fit_lse(fwd_quad, log_L, sd_alt_grid, delta=DELTA, verbose=False)
    p3b = res3b["params"]
    _print_params("(stage 3' — quad from TRIPLE-anchor)", p3b)
    # Pick whichever yields lower in-sample loss
    rmse_a = res3.get("rmse_logL", float("inf"))
    rmse_b = res3b.get("rmse_logL", float("inf"))
    p3 = p3b if rmse_b < rmse_a else p3
    print(f"  stage 3 vs 3' loss: {rmse_a:.4f} vs {rmse_b:.4f} → "
          f"using {'3 (rep-warm)' if p3 is not p3b else '3 (TRIPLE-anchor)'}")

    K_VALUES = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    print(f"\n[Stage 4] iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(data, p3, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  {'E':>6s}  {'A':>6s}  "
          f"{'B':>9s}  {'α':>5s}  {'β':>5s}  | "
          f"{'lK_r':>6s} {'ρ_r':>6s} {'σ_r':>6s} | "
          f"{'lK_p':>6s} {'ρ_p':>6s} {'σ_p':>6s} | "
          f"{'lK_s':>6s} {'ρ_s':>6s} {'σ_s':>6s} | "
          f"{'1ep':>5s}  {'rep':>5s}  {'par':>5s}  {'sd':>5s}")
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(data, r["params"], r["pred_full"], r["keep"])
        p = r["params"]
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  "
              f"{np.exp(p['e']):>6.2f}  {np.exp(p['a']):>6.1f}  "
              f"{np.exp(p['b']):>9.0f}  {p['alpha']:>5.3f}  "
              f"{p['beta']:>5.3f}  | "
              f"{p['log_K_rep']:>6.2f} {p['rho_rep']:>+6.2f} "
              f"{p['sigma_rep']:>+6.2f} | "
              f"{p['log_K_para']:>6.2f} {p['rho_para']:>+6.2f} "
              f"{p['sigma_para']:>+6.2f} | "
              f"{p['log_K_sd']:>6.2f} {p['rho_sd']:>+6.2f} "
              f"{p['sigma_sd']:>+6.2f} | "
              f"{s['rmse_1ep']:>5.3f}  {s['rmse_rep']:>5.3f}  "
              f"{s['rmse_para']:>5.3f}  {s['rmse_sd']:>5.3f}")

    # Require β plateaued AND not still creeping; pick first k where β
    # has plateaued and hasn't gained > 0.005 over the next two k steps either.
    betas = [sweep[k]["params"]["beta"] for k in K_VALUES]
    canonical = K_VALUES[-1]
    for i in range(1, len(betas) - 2):
        plateau = abs(betas[i] - betas[i - 1]) < 0.01
        # also require β not to keep climbing meaningfully afterward
        future = betas[min(i + 2, len(betas) - 1)] - betas[i]
        if plateau and future < 0.02:
            canonical = K_VALUES[i]; break
    print(f"\n  Canonical k (|Δβ|<0.01 first break): k = {canonical}")
    rc = sweep[canonical]
    sc = _summary(data, rc["params"], rc["pred_full"], rc["keep"])
    _print_params(f"(canonical k={canonical})", rc["params"])
    print(f"    RMSE — 1ep:{sc['rmse_1ep']:.4f}  rep:{sc['rmse_rep']:.4f}  "
          f"para:{sc['rmse_para']:.4f}  sd:{sc['rmse_sd']:.4f}  "
          f"|  kept:{sc['rmse_kept']:.4f}")

    # ── Comparison vs reference fits ───────────────────────────────
    print("\n" + "=" * 96)
    print("Comparison with previously-reported fits:")
    print("=" * 96)
    print(f"  {'fit':<42s}  {'E':>6s}  {'A':>6s}  {'B':>7s}  "
          f"{'α':>5s}  {'β':>5s}  {'lK_r':>6s} {'lK_p':>6s} {'lK_s':>6s}")
    print(f"  {'-'*42:<42s}  {'-'*6:>6s}  {'-'*6:>6s}  {'-'*7:>7s}  "
          f"{'-'*5:>5s}  {'-'*5:>5s}  {'-'*6:>6s} {'-'*6:>6s} {'-'*6:>6s}")
    print(f"  {'two-stage 1-ep only (k=20)':<42s}  "
          f"{REF_1EP_ONLY['E']:>6.2f}  {REF_1EP_ONLY['A']:>6.0f}  "
          f"{REF_1EP_ONLY['B']:>7.0f}  {REF_1EP_ONLY['alpha']:>5.3f}  "
          f"{REF_1EP_ONLY['beta']:>5.3f}  {'-':>6s} {'-':>6s} {'-':>6s}")
    print(f"  {'one-shot rep+1ep (k=15, writeup_final)':<42s}  "
          f"{REF_ONESHOT_REP['E']:>6.2f}  {REF_ONESHOT_REP['A']:>6.1f}  "
          f"{REF_ONESHOT_REP['B']:>7.0f}  {REF_ONESHOT_REP['alpha']:>5.3f}  "
          f"{REF_ONESHOT_REP['beta']:>5.3f}  "
          f"{REF_ONESHOT_REP['log_K_rep']:>6.2f} {'-':>6s} {'-':>6s}")
    rt = REF_TRIPLE_K15
    print(f"  {'TRIPLE rep+para+1ep (k=15, §6)':<42s}  "
          f"{rt['E']:>6.3f}  {rt['A']:>6.1f}  "
          f"{rt['B']:>7.0f}  {rt['alpha']:>5.3f}  "
          f"{rt['beta']:>5.3f}  {rt['log_K_rep']:>6.2f} "
          f"{rt['log_K_para']:>6.2f} {'-':>6s}")
    pc = rc["params"]
    print(f"  {f'QUAD all 4 sources (k={canonical}, this script)':<42s}  "
          f"{np.exp(pc['e']):>6.3f}  {np.exp(pc['a']):>6.1f}  "
          f"{np.exp(pc['b']):>7.0f}  {pc['alpha']:>5.3f}  "
          f"{pc['beta']:>5.3f}  {pc['log_K_rep']:>6.2f} "
          f"{pc['log_K_para']:>6.2f} {pc['log_K_sd']:>6.2f}")

    plot_quad_diagnostic(
        data, rc["pred_full"], rc["keep"], rc["dropped"],
        rc["params"], sc,
        path=os.path.join(SCRIPT_DIR, "fit_joint_quad.pdf"))

    return sweep, canonical


if __name__ == "__main__":
    main()
