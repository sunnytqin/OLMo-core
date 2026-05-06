"""
Joint fit on **1-epoch + paraphrase only** (no repetition).

Functional form (8 parameters):

    L(N, D, D') = E + A/N^α + B / (D + η_para(D, D'; N) · D')^β

with the same Form-B exp-sat η:

    η_para(D, D'; N) = R*_para · (1 − e^{−x/R*}) / x,   x = D'/D
    log R*_para = log K_para + ρ_para · log(D/N) + σ_para · log N

For 1-epoch points D' = 0 → η inert.  Five Chinchilla parameters
(E, A, B, α, β) plus three η_para parameters.

Pipeline mirrors fit_joint_triple.py:
  Stage 1: Hoffmann-style 256-init grid on the 8-param model.
  Stage 2: Iterative residual-greedy drop on pooled (1-ep + para).

Usage:
    python fit_joint_para_1ep.py
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
})

from data import (SIZES, extract_1epoch, extract_paraphrase,  # noqa: E402
                  load_with_para)
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.1
SOURCE_NONE, SOURCE_PARA = 0, 1

# Soft caps to keep the optimizer out of the η≈1 degenerate basin
LOG_K_CAP = 10.0   # log_K_para ≤ ≈10 (softplus-smoothed)
RHO_CAP = 2.0      # |ρ_para| ≤ ≈2
SIGMA_CAP = 1.0    # |σ_para| ≤ ≈1


def _soft_cap(x, lo, hi):
    """Smooth clamp x ∈ ℝ to (lo, hi). Differentiable, monotone."""
    span = (hi - lo) / 2.0
    mid = (hi + lo) / 2.0
    return mid + span * torch.tanh((x - mid) / span)


def _bound_eta_params(log_K_raw, rho_raw, sigma_raw):
    log_K = LOG_K_CAP - torch.nn.functional.softplus(LOG_K_CAP - log_K_raw)
    rho = _soft_cap(rho_raw, -RHO_CAP, RHO_CAP)
    sigma = _soft_cap(sigma_raw, -SIGMA_CAP, SIGMA_CAP)
    return log_K, rho, sigma


def bound_params_dict(p):
    """Convert raw fitted params to a dict where η-params are the bounded
    (effective) values for printing / downstream use."""
    out = dict(p)
    log_K, rho, sigma = _bound_eta_params(
        torch.tensor(p["log_K_para"], dtype=torch.float64),
        torch.tensor(p["rho_para"], dtype=torch.float64),
        torch.tensor(p["sigma_para"], dtype=torch.float64))
    out["log_K_para"] = float(log_K)
    out["rho_para"] = float(rho)
    out["sigma_para"] = float(sigma)
    return out


def _Rstar(log_K_raw, rho_raw, sigma_raw, log_DoverN, log_N):
    log_K, rho, sigma = _bound_eta_params(log_K_raw, rho_raw, sigma_raw)
    return torch.exp(log_K + rho * log_DoverN + sigma * log_N)


def _eta_formB(R, x_safe):
    return R * (1.0 - torch.exp(-x_safe / R)) / x_safe


def make_forward(N_arr, D_arr, Dp_arr, source_arr):
    log_N = torch.tensor(np.log(N_arr), dtype=torch.float64)
    D_t  = torch.tensor(D_arr, dtype=torch.float64)
    Dp_t = torch.tensor(Dp_arr, dtype=torch.float64)
    src_t = torch.tensor(source_arr, dtype=torch.int64)
    log_DoverN = torch.log(D_t / torch.tensor(N_arr, dtype=torch.float64))
    is_para = (src_t == SOURCE_PARA)
    safe_D = torch.where(D_t > 0, D_t, torch.ones_like(D_t))
    x = Dp_t / safe_D
    safe_x = torch.where(is_para, x, torch.ones_like(x))

    def forward(p):
        R = _Rstar(p["log_K_para"], p["rho_para"], p["sigma_para"],
                    log_DoverN, log_N)
        eta = _eta_formB(R, safe_x)
        eta = torch.where(is_para, eta, torch.zeros_like(eta))
        D_eff = D_t + eta * Dp_t
        log_D_eff = torch.log(D_eff)
        terms = torch.stack([
            p["e"].expand_as(log_N),
            p["a"] - p["alpha"] * log_N,
            p["b"] - p["beta"] * log_D_eff,
        ], dim=0)
        return logsumexp_stable(terms, dim=0)

    return forward


EXCLUDE_SIZES = {"14m"}  # 14m too small / noisy for joint fit


def collect_pooled():
    tags, Ns, Ds, Dps, Ls, src = [], [], [], [], [], []
    for size in SIZES:
        if size in EXCLUDE_SIZES:
            continue
        N, datasets, parap = load_with_para(size)
        s1, D1, L1 = extract_1epoch(datasets, N, scale_min=0.0)
        for d, l in zip(D1, L1):
            tags.append(size); Ns.append(N); Ds.append(d); Dps.append(0.0)
            Ls.append(l); src.append(SOURCE_NONE)
        if parap:
            sp, Dp_, Kp_, Dpp, Lp, _ = extract_paraphrase(
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


GRID = expand_grid({
    "e":           [0.0, 1.0],
    "a":           [3.0, 12.0],
    "b":           [5.0, 12.0],
    "alpha":       [0.15, 0.4],
    "beta":        [0.3, 0.5],
    "log_K_para":  [4.0, 8.0],
    "rho_para":    [-0.5, -0.2],
    "sigma_para":  [-0.5, -0.2],
})  # 2^5 · 2 · 2 · 2 = 256 inits — small log K, small |ρ|, neg σ
LOG_K_GUARD = 13.0  # reset to fresh grid if warm-start drifts beyond this


def stage1(data, delta=DELTA):
    fwd = make_forward(data["N"], data["D"], data["Dp"], data["source"])
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    return fit_lse(fwd, log_L, GRID, delta=delta, verbose=False)


def _fit_step(last, fwd, log_L, delta):
    """Warm-start from `last`. Soft caps inside _Rstar keep the optimizer
    out of the η≈1 degenerate basin without further guarding."""
    grid = [{kk: float(vv) for kk, vv in last.items()}]
    return fit_lse(fwd, log_L, grid, delta=delta, verbose=False)


def topk_drop_sweep(data, init_params, k_values, delta=DELTA):
    fwd_full = make_forward(data["N"], data["D"], data["Dp"], data["source"])
    last = init_params
    keep = np.ones(len(data["L"]), dtype=bool)
    cumulative = []
    out = {}
    for k in sorted(set(k_values)):
        while len(cumulative) < k:
            fwd = make_forward(
                data["N"][keep], data["D"][keep],
                data["Dp"][keep], data["source"][keep])
            log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
            res = _fit_step(last, fwd, log_L, delta)
            with torch.no_grad():
                p_t = {kk: torch.tensor(v, dtype=torch.float64)
                       for kk, v in res["params"].items()}
                pred_full = fwd_full(p_t).numpy()
            resid = np.log(data["L"]) - pred_full
            ar = np.abs(resid); ar[~keep] = -1.0
            worst = int(np.argmax(ar))
            cumulative.append(worst); keep[worst] = False
            last = res["params"]
        fwd = make_forward(
            data["N"][keep], data["D"][keep],
            data["Dp"][keep], data["source"][keep])
        log_L = torch.tensor(np.log(data["L"][keep]), dtype=torch.float64)
        res = _fit_step(last, fwd, log_L, delta)
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
    for tag, src_id in [("1ep", SOURCE_NONE), ("para", SOURCE_PARA)]:
        m = (data["source"] == src_id)
        mk = m & keep
        summary[f"n_{tag}"] = int(m.sum())
        summary[f"n_{tag}_kept"] = int(mk.sum())
        summary[f"rmse_{tag}"] = (float(np.sqrt(np.mean(resid[mk] ** 2)))
                                   if mk.any() else float("nan"))
    summary["params"] = params
    return summary


def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    return f"{x:.0f}"


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
        mp = m & (data["source"] == SOURCE_PARA)
        ax_data.scatter(data["D"][mp] + data["Dp"][mp], data["L"][mp],
                        s=42, color=c, edgecolors="k", linewidths=0.5,
                        marker="s", alpha=0.85, zorder=3)
        for src_id, mk_marker in [(SOURCE_NONE, "o"), (SOURCE_PARA, "s")]:
            mm = m & (data["source"] == src_id)
            if mm.any():
                ax_res.scatter(data["D"][mm], resid[mm], s=40, color=c,
                               edgecolors="k", linewidths=0.3, marker=mk_marker)
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
    p = bound_params_dict(params)
    ax_data.set_title(
        rf"1ep+para fit: $E={np.exp(p['e']):.3f}$, $A={np.exp(p['a']):.0f}$, "
        rf"$B={np.exp(p['b']):.0f}$, $\alpha={p['alpha']:.3f}$, "
        rf"$\beta={p['beta']:.3f}$" + "\n"
        rf"η_para: $\log K={p['log_K_para']:.2f}$, "
        rf"$\rho={p['rho_para']:+.3f}$, $\sigma={p['sigma_para']:+.3f}$",
        fontsize=11)

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_res.set_xscale("log", base=2)
    ax_res.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_res.set_xlabel(r"$D$")
    ax_res.set_ylabel(r"residual ($\log L$)")
    ax_res.set_title(rf"Residuals  (kept RMSE: 1ep={summary['rmse_1ep']:.3f}, "
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
        "Joint fit on 1-epoch + paraphrase  (○ 1ep, □ para)",
        fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _print_params(label, p):
    pb = bound_params_dict(p)
    print(f"  {label}")
    print(f"    Chinchilla: E={np.exp(pb['e']):.4f}  A={np.exp(pb['a']):.2f}  "
          f"B={np.exp(pb['b']):.2f}  α={pb['alpha']:.4f}  β={pb['beta']:.4f}")
    print(f"    η_para:     log K={pb['log_K_para']:.2f}  "
          f"ρ={pb['rho_para']:+.3f}  σ={pb['sigma_para']:+.3f}  "
          f"(caps: log_K≤{LOG_K_CAP}, |ρ|≤{RHO_CAP}, |σ|≤{SIGMA_CAP})")


def main():
    print("=" * 96)
    print("Joint fit: 1-epoch + paraphrase only  "
          "(8 params: shared E, A, B, α, β + η_para)")
    print("=" * 96)

    data = collect_pooled()
    n_per = {SOURCE_NONE: int((data["source"] == SOURCE_NONE).sum()),
             SOURCE_PARA: int((data["source"] == SOURCE_PARA).sum())}
    print(f"\nPooled: n_total={len(data['L'])}  "
          f"(1ep={n_per[SOURCE_NONE]}, para={n_per[SOURCE_PARA]})")

    print(f"\n[Stage 1] Hoffmann-style grid fit ({len(GRID)} inits)...")
    res1 = stage1(data)
    p1 = res1["params"]
    print(f"  Stage-1 RMSE (logL): {res1.get('rmse_logL', float('nan')):.4f}")
    _print_params("(stage 1 — full pool)", p1)

    K_VALUES = [0, 5, 10, 15, 20, 25, 30]
    print(f"\n[Stage 2] Iterative residual drop, k ∈ {K_VALUES}...")
    sweep = topk_drop_sweep(data, p1, K_VALUES)

    print(f"\n  {'k':>3s}  {'n_kept':>6s}  {'E':>7s}  {'A':>7s}  "
          f"{'B':>9s}  {'α':>6s}  {'β':>6s}  | "
          f"{'lK_par':>7s} {'ρ_par':>7s} {'σ_par':>7s} | "
          f"{'1ep':>5s}  {'par':>5s}")
    for k in K_VALUES:
        r = sweep[k]
        s = _summary(data, r["params"], r["pred_full"], r["keep"])
        p = bound_params_dict(r["params"])
        print(f"  {k:>3d}  {int(r['keep'].sum()):>6d}  "
              f"{np.exp(p['e']):>7.3f}  {np.exp(p['a']):>7.1f}  "
              f"{np.exp(p['b']):>9.0f}  {p['alpha']:>6.3f}  "
              f"{p['beta']:>6.3f}  | "
              f"{p['log_K_para']:>7.2f} {p['rho_para']:>+7.3f} "
              f"{p['sigma_para']:>+7.3f} | "
              f"{s['rmse_1ep']:>5.3f}  {s['rmse_para']:>5.3f}")

    betas = [sweep[k]["params"]["beta"] for k in K_VALUES]
    canonical = K_VALUES[-1]
    for i in range(1, len(betas)):
        if abs(betas[i] - betas[i - 1]) < 0.01:
            canonical = K_VALUES[i]; break
    print(f"\n  Canonical k (|Δβ|<0.01 first break): k = {canonical}")
    rc = sweep[canonical]
    sc = _summary(data, rc["params"], rc["pred_full"], rc["keep"])
    _print_params(f"(canonical k={canonical})", rc["params"])
    print(f"    fit RMSE — 1ep: {sc['rmse_1ep']:.4f}  "
          f"para: {sc['rmse_para']:.4f}  |  kept-total: {sc['rmse_kept']:.4f}")

    plot_diagnostic(
        data, rc["pred_full"], rc["keep"], rc["dropped"],
        rc["params"], sc,
        path=os.path.join(SCRIPT_DIR, "fit_joint_para_1ep.pdf"))

    return sweep, canonical


if __name__ == "__main__":
    main()
