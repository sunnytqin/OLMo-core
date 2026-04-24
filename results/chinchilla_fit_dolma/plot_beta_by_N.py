"""
Verify whether β genuinely varies with N, or is just fit noise.

Two panels:
  (A) Log-log plot of (L − E_eff(N)) vs D for each size, with a
      straight-line fit through that size's points.  Slope = local β.
      On a single joint β, all lines should be parallel.
  (B) Fitted β per size, three estimates:
        - joint 5-param Chinchilla (shared β)
        - per-size 3-param Chinchilla (independent)
        - per-size 2-param fit with E_eff fixed from joint (most conservative)
      Plotted vs log N with the joint β as a horizontal reference.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))
from data import SIZES, load, extract_1epoch, DEFAULT_SCALE_MIN, TTP_RATIO  # noqa
from fit_lse import expand_grid, fit_lse, logsumexp_stable  # noqa
from fit_chinchilla_joint import collect_1epoch_all_sizes, fit_joint, DELTA  # noqa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALE_MIN = DEFAULT_SCALE_MIN

FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 15, 13, 11, 17


# ──────────────────────────────────────────────────────────────────────

def per_size_3p_fit(scale, D, L, delta=DELTA):
    """Fit L = E + B/D^β (3 params) on a single size's 1-epoch points."""
    grid = expand_grid({
        "e":    [-1.0, 0.0, 0.5, 1.0, 1.5],
        "b":    [0.0, 5.0, 10.0, 15.0],
        "beta": [0.1, 0.3, 0.5, 0.7, 1.0],
    })
    log_D = torch.tensor(np.log(D), dtype=torch.float64)
    log_L = torch.tensor(np.log(L), dtype=torch.float64)

    def forward(p):
        terms = torch.stack([p["e"].expand_as(log_D),
                             p["b"] - p["beta"] * log_D], dim=0)
        return logsumexp_stable(terms, dim=0)

    res = fit_lse(forward, log_L, grid, delta=delta)
    p = res["params"]
    return dict(E=float(np.exp(p["e"])), B=float(np.exp(p["b"])),
                beta=float(p["beta"]))


def per_size_2p_fit(D, L, E_fixed, delta=DELTA):
    """Fit L = E_fixed + B/D^β with E clamped from joint; 2 params."""
    grid = expand_grid({
        "b":    [0.0, 5.0, 10.0, 15.0, 20.0],
        "beta": [0.05, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0],
    })
    log_D = torch.tensor(np.log(D), dtype=torch.float64)
    log_L = torch.tensor(np.log(L), dtype=torch.float64)
    e_c = torch.tensor(float(np.log(E_fixed)), dtype=torch.float64)

    def forward(p):
        terms = torch.stack([e_c.expand_as(log_D),
                             p["b"] - p["beta"] * log_D], dim=0)
        return logsumexp_stable(terms, dim=0)

    res = fit_lse(forward, log_L, grid, delta=delta)
    p = res["params"]
    return dict(E=float(E_fixed), B=float(np.exp(p["b"])),
                beta=float(p["beta"]))


def log_log_slope(D, y):
    """OLS slope on log-log with no error bars; returns slope."""
    x = np.log(D); z = np.log(y)
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, z, rcond=None)[0]
    return m, c


# ──────────────────────────────────────────────────────────────────────

def fmt_tokens(x, pos=None):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def main():
    # ── joint fit ──
    tags, N_arr, scale_arr, D_arr, L_arr = collect_1epoch_all_sizes(SCALE_MIN)
    anchor = fit_joint(N_arr, D_arr, L_arr, delta=DELTA)
    beta_joint = anchor["beta"]
    sizes = sorted(set(tags.tolist()), key=lambda t: SIZES[t][0])
    N_by_size = {t: SIZES[t][0] for t in sizes}

    # ── per-size fits ──
    fit_3p, fit_2p, E_eff_joint = {}, {}, {}
    print(f"{'size':<6s}  {'n':>3s}  {'β_3p':>7s}  {'β_2p':>7s}  "
          f"{'β_joint':>8s}  {'E_eff(N)':>9s}")
    for tag in sizes:
        m = tags == tag
        D, L, sc = D_arr[m], L_arr[m], scale_arr[m]
        if len(D) < 3:
            continue
        E_eff = anchor["E"] + anchor["A"] / N_by_size[tag] ** anchor["alpha"]
        f3 = per_size_3p_fit(sc, D, L)
        f2 = per_size_2p_fit(D, L, E_fixed=E_eff)
        fit_3p[tag] = f3
        fit_2p[tag] = f2
        E_eff_joint[tag] = E_eff
        print(f"{tag:<6s}  {len(D):>3d}  {f3['beta']:>7.4f}  "
              f"{f2['beta']:>7.4f}  {beta_joint:>8.4f}  {E_eff:>9.4f}")

    # ── plot ──
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    ax_ll, ax_bN = axes

    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=len(sizes) - 1)

    # (A) log-log: (L − E_eff) vs D, per size, with local slope
    for i, tag in enumerate(sizes):
        m = tags == tag
        if tag not in fit_2p:
            continue
        D, L = D_arr[m], L_arr[m]
        color = cmap(cnorm(i))
        resid = L - E_eff_joint[tag]
        valid = resid > 0
        ax_ll.scatter(D[valid], resid[valid], s=70, color=color,
                      edgecolors="k", linewidths=0.4, zorder=5,
                      label=f"{tag}  β_2p={fit_2p[tag]['beta']:.3f}")
        # 2-param fit curve (uses joint E_eff)
        D_smooth = np.geomspace(D[valid].min() * 0.7, D[valid].max() * 1.3, 80)
        y_smooth = fit_2p[tag]["B"] / D_smooth ** fit_2p[tag]["beta"]
        ax_ll.plot(D_smooth, y_smooth, "-", color=color, linewidth=1.6, alpha=0.85)

    # Joint β reference slope, normalized at the median D
    D_med = np.exp(np.median(np.log(D_arr)))
    y_med = (L_arr - np.array([E_eff_joint[t] for t in tags])).mean()
    ref_D = np.geomspace(D_arr.min() * 0.8, D_arr.max() * 1.2, 80)
    ref_y = y_med * (ref_D / D_med) ** (-beta_joint)
    ax_ll.plot(ref_D, ref_y, "--", color="black", linewidth=1.5, alpha=0.5,
               label=f"joint β = {beta_joint:.3f}")

    ax_ll.set_xscale("log", base=2)
    ax_ll.set_yscale("log")
    ax_ll.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_ll.set_xlabel("Training tokens D", fontsize=FONT_LABEL)
    ax_ll.set_ylabel(r"$L - E_{\mathrm{eff}}(N)$   (log)", fontsize=FONT_LABEL)
    ax_ll.set_title("Per-size power-law diagnostic\n(parallel lines ⇒ shared β)",
                    fontsize=FONT_TITLE)
    ax_ll.tick_params(labelsize=FONT_TICK)
    ax_ll.grid(alpha=0.3, which="both")
    ax_ll.legend(fontsize=FONT_LEGEND - 1, loc="best")

    # (B) β vs N — three estimates
    Ns = np.array([N_by_size[t] for t in sizes])
    betas_3p = np.array([fit_3p[t]["beta"] for t in sizes])
    betas_2p = np.array([fit_2p[t]["beta"] for t in sizes])

    ax_bN.axhline(beta_joint, color="black", linestyle="--", linewidth=1.5,
                  alpha=0.5, label=f"joint β = {beta_joint:.3f}")
    ax_bN.plot(Ns, betas_3p, "o-", color="tab:orange", markersize=10,
               linewidth=1.8, label="per-size 3-param fit  (E, B, β)", zorder=5)
    ax_bN.plot(Ns, betas_2p, "s-", color="tab:blue", markersize=10,
               linewidth=1.8, label="per-size 2-param fit  (B, β | E_eff from joint)",
               zorder=5)

    # Annotate each size
    for t, N, b3 in zip(sizes, Ns, betas_3p):
        ax_bN.annotate(t, (N, b3), textcoords="offset points",
                       xytext=(8, 8), fontsize=FONT_LEGEND, color="gray")

    ax_bN.set_xscale("log")
    ax_bN.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_bN.set_xlabel("Model parameters N", fontsize=FONT_LABEL)
    ax_bN.set_ylabel(r"fitted $\beta$", fontsize=FONT_LABEL)
    ax_bN.set_title(r"$\beta$ vs N", fontsize=FONT_TITLE)
    ax_bN.tick_params(labelsize=FONT_TICK)
    ax_bN.grid(alpha=0.3)
    ax_bN.legend(fontsize=FONT_LEGEND, loc="best")

    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, "plot_beta_by_N.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
