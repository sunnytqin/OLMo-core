"""
Forest plot of the bootstrap 95% CIs for the one-go triple-fit parameters.

Reads:
  _onego_json/bootstrap_{all,drop14m}_B200_seed42.json

Two output modes (controlled by --layout):
  4panel : grouped 4-panel figure comparing all-data vs drop14m
  rows   : single-column figure with 11 row-subplots for ONE variant

Examples:
  python bootstrap_forest.py --layout 4panel --out bootstrap_forest.pdf
  python bootstrap_forest.py --layout rows --variant drop14m \\
      --out bootstrap_forest_drop14m.pdf
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "xtick.labelsize":  10,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.titlesize": 14,
})

VARIANTS = ["all", "drop14m"]
COLOR = {"all": "#1f77b4", "drop14m": "#d62728"}
LABEL = {"all": "all data (n=320 kept of 335)",
         "drop14m": "drop 14M non-para (n=271 kept of 286)"}

PARAMS_ROWS = [
    # (json_key, latex_label, exp_format)
    ("e",           r"$\log E$",                "exp"),
    ("a",           r"$\log A$",                "exp"),
    ("b",           r"$\log B$",                "exp"),
    ("alpha",       r"$\alpha$",                None),
    ("beta",        r"$\beta$",                 None),
    ("log_K_rep",   r"$\log K_{\mathrm{rep}}$", None),
    ("rho_rep",     r"$\rho_{\mathrm{rep}}$",   None),
    ("sigma_rep",   r"$\sigma_{\mathrm{rep}}$", None),
    ("log_K_para",  r"$\log K_{\mathrm{para}}$",None),
    ("rho_para",    r"$\rho_{\mathrm{para}}$",  None),
    ("sigma_para",  r"$\sigma_{\mathrm{para}}$",None),
]
GROUP_OF = {
    "e": "Chinchilla", "a": "Chinchilla", "b": "Chinchilla",
    "alpha": "Chinchilla", "beta": "Chinchilla",
    "log_K_rep": "eta_rep", "rho_rep": "eta_rep", "sigma_rep": "eta_rep",
    "log_K_para": "eta_para", "rho_para": "eta_para",
    "sigma_para": "eta_para",
}
GROUP_COLORS = {
    "Chinchilla":  "#1f77b4",
    "eta_rep":     "#2ca02c",
    "eta_para":    "#d62728",
}


def load(variant):
    p = os.path.join(SCRIPT_DIR, "_onego_json",
                      f"bootstrap_{variant}_B200_seed42.json")
    with open(p) as f:
        return json.load(f)


def forest_row(ax, y, summary, color, param, offset=0.0):
    s = summary[param]
    med, lo, hi = s["median"], s["ci_2_5"], s["ci_97_5"]
    ax.errorbar([med], [y + offset],
                xerr=[[med - lo], [hi - med]],
                fmt="o", color=color, ecolor=color, capsize=4, lw=1.6,
                markersize=7, markeredgecolor="k", markeredgewidth=0.5)


def plot_4panel(out):
    runs = {v: load(v) for v in VARIANTS}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    (ax_chi_const, ax_chi_exp), (ax_logK, ax_eta) = axes
    delta = 0.18

    panels = [
        (ax_chi_const, [("e", r"$\log E$"), ("a", r"$\log A$"),
                         ("b", r"$\log B$")],
          "Chinchilla constants (log-space)", True),
        (ax_chi_exp, [("alpha", r"$\alpha$"), ("beta", r"$\beta$")],
          r"Chinchilla exponents", False),
        (ax_logK, [("log_K_rep", r"$\log K_{\mathrm{rep}}$"),
                     ("log_K_para", r"$\log K_{\mathrm{para}}$")],
          r"$\log K$ (rep vs para)", False),
        (ax_eta, [("rho_para", r"$\rho_{\mathrm{para}}$"),
                    ("sigma_para", r"$\sigma_{\mathrm{para}}$"),
                    ("rho_rep", r"$\rho_{\mathrm{rep}}$"),
                    ("sigma_rep", r"$\sigma_{\mathrm{rep}}$")],
          r"η exponents ($\rho$, $\sigma$): zero-line shown for sign analysis",
          False),
    ]
    for ax, params, title, has_e_anno in panels:
        for i, (p, _) in enumerate(params):
            for j, v in enumerate(VARIANTS):
                forest_row(ax, len(params) - i - 1, runs[v]["summary"],
                            COLOR[v], p, offset=delta * (j - 0.5) * 2)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([lbl for _, lbl in params[::-1]])
        ax.set_xlabel("value" if not has_e_anno else "value (log-space)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.set_ylim(-0.5, len(params) - 0.5)

    ax_chi_const.axvline(0, color="0.6", linestyle="--", linewidth=0.8,
                          alpha=0.5)
    ax_eta.axvline(0, color="0.4", linestyle="-", linewidth=1.4, alpha=0.8,
                    zorder=0)

    # Exp annotations on Chinchilla constants panel
    for i, p in enumerate(["e", "a", "b"]):
        med_all = runs["all"]["summary"][p]["median"]
        med_drop = runs["drop14m"]["summary"][p]["median"]
        ax_chi_const.text(
            0.02, 0.98 - i * 0.16,
            f"exp({p}): all={np.exp(med_all):.2g}, "
            f"drop14m={np.exp(med_drop):.2g}",
            transform=ax_chi_const.transAxes, fontsize=9,
            verticalalignment="top", color="0.3", fontfamily="serif")

    handles = [Line2D([0], [0], marker="o", linestyle="", color=COLOR[v],
                       markersize=7, markeredgecolor="k",
                       markeredgewidth=0.5, label=LABEL[v]) for v in VARIANTS]
    ax_eta.legend(handles=handles, loc="upper right", fontsize=9,
                   framealpha=0.95)
    fig.suptitle(
        r"One-go triple fit — 95% bootstrap CIs ($B=200$, parametric "
        r"resample of canonical $k=15$ kept set)",
        fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_rows(variant, out):
    d = load(variant)
    summary = d["summary"]
    n_kept = d["n_kept"]; n_full = d["n_full"]; B = d["B_succeeded"]

    n = len(PARAMS_ROWS)
    fig = plt.figure(figsize=(9.5, 0.9 * n + 1.2))
    gs = GridSpec(n, 1, figure=fig,
                   left=0.18, right=0.78, top=0.93, bottom=0.07,
                   hspace=0.45)
    for i, (key, label, fmt) in enumerate(PARAMS_ROWS):
        s = summary[key]
        med, lo, hi = s["median"], s["ci_2_5"], s["ci_97_5"]
        c = GROUP_COLORS[GROUP_OF[key]]
        ax = fig.add_subplot(gs[i, 0])
        ci_w = max(hi - lo, 1e-3)
        x_pad = 0.30 * ci_w
        x_lo = lo - x_pad
        x_hi = hi + x_pad
        ax.errorbar([med], [0], xerr=[[med - lo], [hi - med]],
                    fmt="o", color=c, ecolor=c, capsize=5, lw=2.2,
                    markersize=9, markeredgecolor="k",
                    markeredgewidth=0.6, zorder=5)
        if key in ("rho_rep", "sigma_rep", "rho_para", "sigma_para"):
            if x_lo <= 0 <= x_hi or abs(x_lo) < 2 * ci_w:
                x_lo = min(x_lo, -0.05 * (abs(x_hi) + 0.1))
                x_hi = max(x_hi, 0.05)
            ax.axvline(0, color="0.4", linestyle="-", linewidth=1.3,
                        alpha=0.8, zorder=1)
        ax.set_xlim(x_lo, x_hi)
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=13)
        ax.set_ylim(-0.6, 0.6)
        ax.grid(axis="x", alpha=0.3, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", which="both", length=0)
        extra = ""
        if fmt == "exp":
            extra = f"  (exp = {np.exp(med):.3g})"
        ax.text(1.02, 0.5, f"{med:+.3f}  [{lo:+.3f}, {hi:+.3f}]{extra}",
                transform=ax.transAxes, fontsize=10.5,
                verticalalignment="center", horizontalalignment="left",
                color="0.2", fontfamily="monospace")

    fig.suptitle(
        r"One-go triple fit — 95% bootstrap CIs "
        rf"({variant}, $B={B}$, $n_{{\mathrm{{kept}}}}={n_kept}$ / {n_full})",
        fontsize=13.5, y=0.985)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["4panel", "rows"], default="4panel")
    ap.add_argument("--variant", choices=VARIANTS, default="drop14m",
                     help="Used only when --layout rows")
    ap.add_argument("--out", default=None,
                     help="Output PDF path (default: auto-named)")
    args = ap.parse_args()

    if args.layout == "4panel":
        out = args.out or os.path.join(SCRIPT_DIR, "bootstrap_forest.pdf")
        plot_4panel(out)
    else:
        out = (args.out or
               os.path.join(SCRIPT_DIR,
                             f"bootstrap_forest_{args.variant}.pdf"))
        plot_rows(args.variant, out)


if __name__ == "__main__":
    main()
