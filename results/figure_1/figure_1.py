"""Figure 1 — three-panel schematic of the multi-epoch CD law.

(a) Single-size (30M) view of the (D, L) plane with three regimes:
    compute-bound, data-bound, model-bound. Bounded by the 1-epoch
    fit, the D'/D -> infinity fit, and the irreducible loss E_eff(N).
(b) All multi-epoch points after D_eff = D + eta(D, D'; N) * D' —
    they collapse onto each size's Chinchilla curve.
(c) Mapping D + D' (total tokens trained) -> D + eta * D' (effective
    fresh-token equivalents). Points stay below y=x; per-size curves
    show diminishing returns from repetition.

Anchors are the canonical one-shot joint fit (k=15, all 7 sizes) from
results/chinchilla_fit_dolma/writeup_final.md.
"""

import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Add ../results so we can import dolma_*.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# === Palatino (URW P052 clone) font setup ===
for _f in glob.glob('/usr/share/fonts/urw-base35/P052-*.otf'):
    font_manager.fontManager.addfont(_f)

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['P052', 'Palatino', 'TeX Gyre Pagella', 'serif'],
    'mathtext.fontset':  'cm',
    'font.size':         12,
    'axes.titlesize':    13,
    'axes.labelsize':    13,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'figure.titlesize':  14,
    'lines.linewidth':   1.8,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '-',
    'savefig.bbox':      'tight',
    'savefig.dpi':       300,
})

# === Scaling-law parameters (writeup_final.md, one-shot joint fit, k=15) ===
SL_E     = 0.050
SL_A     = 31.5
SL_B     = 16539.0
SL_ALPHA = 0.137
SL_BETA  = 0.436
SL_LOG_K = 10.32
SL_RHO   = -0.270
SL_SIGMA = -0.388

TTP_RATIO = 20  # Chinchilla tokens-per-param


def E_eff(N):
    return SL_E + SL_A / (N ** SL_ALPHA)


def L_chin(N, D_eff):
    D_eff = np.asarray(D_eff, dtype=float)
    return E_eff(N) + SL_B / D_eff ** SL_BETA


def R_star(D, N):
    D = np.asarray(D, dtype=float)
    return np.exp(SL_LOG_K + SL_RHO * np.log(D / N) + SL_SIGMA * np.log(N))


def eta_of(D, Dp, N):
    """eta(D, D'; N) = R*(1-exp(-x/R*))/x, x = D'/D. eta(0)=1 limit."""
    D = np.asarray(D, dtype=float)
    Dp = np.asarray(Dp, dtype=float)
    R = R_star(D, N)
    x = np.where(Dp > 0, Dp / D, 1.0)
    val = np.where(Dp > 0, R * (1.0 - np.exp(-x / R)) / x, 1.0)
    return val


def L_inf_epochs(D, N):
    """Loss with D'/D -> infinity (multi-epoch limit). D_eff = D*(1 + R*)."""
    D = np.asarray(D, dtype=float)
    return E_eff(N) + SL_B / (D * (1.0 + R_star(D, N))) ** SL_BETA


def L_iso_x(D, x, N):
    """CD-law loss at fixed D'/D = x (epochs = 1 + x)."""
    D = np.asarray(D, dtype=float)
    if x == 0:
        return L_chin(N, D)
    R = R_star(D, N)
    eta_times_x = R * (1.0 - np.exp(-x / R))
    return E_eff(N) + SL_B / (D * (1.0 + eta_times_x)) ** SL_BETA


ISO_X_VALUES = [1, 3, 7, 15, 31, 63]


# === Sizes with multi-epoch coverage ===
SIZES = [
    ('14m',  14e6,  'dolma_14m'),
    ('30m',  30e6,  'dolma_30m'),
    ('60m',  60e6,  'dolma_60m'),
    ('190m', 190e6, 'dolma_190m'),
    ('370m', 370e6, 'dolma_370m'),
]

OVERFIT_EXCLUDE = {
    '14m':  {(0.05, 128), (0.1, 128)},
    '30m':  {(0.05, 128), (0.1, 128)},
    '60m':  {(0.1, 64)},
    '190m': {(0.05, 32), (0.05, 64), (0.1, 32), (0.25, 32), (0.5, 32)},
    '370m': {(0.05, 32), (0.05, 64)},
}


def load_points(tag, modname, N):
    mod = __import__(modname)
    rows = []
    excl = OVERFIT_EXCLUDE.get(tag, set())
    for ds in mod.ALL_DATASETS:
        scale = ds['chinchilla_scale'][0]
        D = scale * TTP_RATIO * N
        for ep, loss, fmul in zip(ds['epochs'], ds['validation_loss'],
                                   ds['flops_multiplier']):
            if np.isnan(loss):
                continue
            if (scale, ep) in excl:
                continue
            Dp = (ep - 1) * D
            rows.append((scale, ep, D, Dp, loss, fmul))
    if not rows:
        return np.empty((0, 6))
    return np.array(rows, dtype=float)


# === Visual constants ===
FONT_LABEL  = 14
FONT_TITLE  = 15
FONT_TICK   = 12
FONT_LEGEND = 11
FONT_REGION = 13
FONT_CBAR   = 12

CMAP = plt.cm.plasma  # for model-size coloring in panel (c)

# Compute / FLOPs colormap (matches results/data_optimal_scaling_iso.py)
CMAP_COMPUTE = LinearSegmentedColormap.from_list(
    'compute',
    ['#A8E6CF', '#5BC4B0', '#3B9AB2', '#2C6E91', '#1F4E79', '#152F4F', '#0A1628'],
)

# Marker sizes for (1-epoch, multi-epoch)
S_1EP   = 110
S_MULTI = 30
S_PANEL_C = 70

# Sizes shown in panel (c)
PANEL_C_TAGS = ['30m', '60m', '370m']

# Region colors (match results/data_optimal_scaling_iso.py for consistency)
COLOR_COMPUTE_REGION = '#A8C5DC'   # soft blue
COLOR_DATA_REGION    = '#E8B4A0'   # muted terracotta
COLOR_MODEL_REGION   = '#C8D5B9'   # sage
ALPHA_COMPUTE = 0.45
ALPHA_DATA    = 0.40
ALPHA_MODEL   = 0.45

COLOR_COMPUTE_DARK = '#1F4E79'
COLOR_DATA_DARK    = '#8B4513'
COLOR_MODEL_DARK   = '#1b5e20'

COLOR_FIT_1EP    = 'black'
COLOR_FIT_MEINF  = '#cb181d'
COLOR_IRREDUCIBLE = '#1b5e20'

# Single anchor size for panel (a)
PANEL_A_SIZE = ('30m', 30e6, 'dolma_30m')

# X-axis lower bound for all three panels
X_MIN = 1e8  # 100M


def size_color(i, n):
    # span 0.10–0.85 of plasma so endpoints aren't pure black or pure yellow
    return CMAP(0.10 + 0.75 * i / max(n - 1, 1))


def fmt_tokens(x, pos=None):
    if x >= 1e9: return f'{x/1e9:g}B'
    if x >= 1e6: return f'{x/1e6:g}M'
    if x >= 1e3: return f'{x/1e3:g}K'
    return f'{x:g}'


def main():
    points = {}
    for tag, N, modname in SIZES:
        points[tag] = (N, load_points(tag, modname, N))

    n_sizes = len(SIZES)
    colors = {tag: size_color(i, n_sizes) for i, (tag, _, _) in enumerate(SIZES)}

    # Shared FLOPs norm across all data
    all_flops = np.concatenate([points[t][1][:, 5] for t, _, _ in SIZES])
    flops_norm = plt.Normalize(vmin=np.log2(all_flops.min()),
                                vmax=np.log2(all_flops.max()))

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.22)

    # ---------- Panel (a): single-size three regimes ----------
    tag_a, N_a, _ = PANEL_A_SIZE
    _, pts_a = points[tag_a]
    scale_a, ep_a, D_a, Dp_a, L_a, F_a = pts_a.T
    is_1ep_a = ep_a == 1
    log2F_a = np.log2(F_a)

    # Curves
    X_MAX_A = 2.0e10
    xa = np.geomspace(X_MIN, X_MAX_A, 500)
    l1ep_xa = L_chin(N_a, xa)
    lme_xa  = L_inf_epochs(xa, N_a)
    e_eff_a = E_eff(N_a)

    # Y-range: capped at 8.5 to reduce white space.
    L_min_a = min(L_a.min(), e_eff_a) - 0.4
    L_max_a = 8.5
    FLOOR = e_eff_a - 0.5

    # Region fills (no legend entries — annotated in-plot)
    ax_a.fill_between(xa, lme_xa, l1ep_xa, where=l1ep_xa > lme_xa,
                      color=COLOR_COMPUTE_REGION, alpha=ALPHA_COMPUTE,
                      zorder=2)
    ax_a.fill_between(xa, e_eff_a, lme_xa, where=lme_xa > e_eff_a,
                      color=COLOR_DATA_REGION, alpha=ALPHA_DATA,
                      zorder=3)
    ax_a.fill_between([X_MIN, X_MAX_A], FLOOR, e_eff_a,
                      color=COLOR_MODEL_REGION, alpha=ALPHA_MODEL,
                      zorder=1)

    # Iso-D'/D dashed curves coloured by FLOPs (CD-law predictions)
    for x in ISO_X_VALUES:
        L_xi = L_iso_x(xa, x, N_a)
        flops_xa = (xa / (TTP_RATIO * N_a)) * (1.0 + x)
        log2_flops = np.log2(flops_xa)
        pts = np.array([xa, L_xi]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=CMAP_COMPUTE, norm=flops_norm,
                            linestyles='dashed', linewidths=1.4,
                            alpha=0.95, zorder=11)
        lc.set_array(0.5 * (log2_flops[:-1] + log2_flops[1:]))
        ax_a.add_collection(lc)

    # Boundary curves (labels handled by text annotations below).
    ax_a.plot(xa, l1ep_xa, '-', color=COLOR_FIT_1EP, linewidth=2.5,
              zorder=12)
    ax_a.plot(xa, lme_xa, '-', color=COLOR_FIT_MEINF, linewidth=2.5,
              zorder=12)
    ax_a.axhline(e_eff_a, linestyle='-', color=COLOR_IRREDUCIBLE,
                 linewidth=2.5, zorder=12)

    # "compute-optimal scaling": text in upper-left between 100M and 1B,
    # dashed arrow pointing mostly leftward onto the black curve.
    x_co_text, y_co_text = 2.2e8, L_max_a - 0.4
    x_co_tip = 1.2e8
    y_co_tip = float(L_chin(N_a, x_co_tip))
    ax_a.annotate('compute-optimal scaling',
                  xy=(x_co_tip, y_co_tip),
                  xytext=(x_co_text, y_co_text),
                  fontsize=FONT_REGION, ha='left', va='center',
                  color=COLOR_FIT_1EP, fontstyle='italic', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', linestyle='dashed',
                                  color=COLOR_FIT_1EP, lw=1.3,
                                  shrinkA=2, shrinkB=2),
                  zorder=15)

    # "data-optimal scaling": text at far left of the data-bound region,
    # dashed arrow pointing right-and-up onto the red curve.
    x_do_text = 1.15e8
    y_do_text = e_eff_a + 0.20
    x_do_tip = 5e8
    y_do_tip = float(L_inf_epochs(x_do_tip, N_a))
    ax_a.annotate('data-optimal scaling',
                  xy=(x_do_tip, y_do_tip),
                  xytext=(x_do_text, y_do_text),
                  fontsize=FONT_REGION, ha='left', va='center',
                  color=COLOR_FIT_MEINF, fontstyle='italic', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', linestyle='dashed',
                                  color=COLOR_FIT_MEINF, lw=1.3,
                                  shrinkA=2, shrinkB=2),
                  zorder=15)

    # Data points (30M only). 1-ep large, multi-ep smaller; coloured by FLOPs.
    ax_a.scatter(D_a[~is_1ep_a], L_a[~is_1ep_a], s=S_MULTI,
                 c=log2F_a[~is_1ep_a], cmap=CMAP_COMPUTE, norm=flops_norm,
                 edgecolors='black', linewidths=0.7,
                 zorder=10, marker='o')
    sc_a = ax_a.scatter(D_a[is_1ep_a], L_a[is_1ep_a], s=S_1EP,
                        c=log2F_a[is_1ep_a], cmap=CMAP_COMPUTE,
                        norm=flops_norm, edgecolors='black', linewidths=0.9,
                        zorder=11, marker='o')

    # Region labels.  Compute-bound: text above + right of the band, with
    # a diagonal dashed arrow pointing into the blue region.
    x_cb_text  = 5.0e9
    y_cb_text  = float(L_chin(N_a, x_cb_text)) + 1.2
    x_cb_arrow = 1.3e9
    y_cb_arrow = 0.5 * (float(L_chin(N_a, x_cb_arrow))
                        + float(L_inf_epochs(x_cb_arrow, N_a)))
    ax_a.annotate('Compute-bound',
                  xy=(x_cb_arrow, y_cb_arrow),
                  xytext=(x_cb_text, y_cb_text),
                  fontsize=FONT_REGION, ha='center', va='bottom',
                  color=COLOR_COMPUTE_DARK,
                  fontstyle='italic', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', linestyle='dashed',
                                  color=COLOR_COMPUTE_DARK, lw=1.5,
                                  shrinkA=4, shrinkB=4),
                  zorder=15)
    x_db = 1.2e10
    ax_a.text(x_db, (float(L_inf_epochs(x_db, N_a)) + e_eff_a) / 2,
              'Data-bound', fontsize=FONT_REGION, ha='center', va='center',
              color=COLOR_DATA_DARK, fontstyle='italic', fontweight='bold',
              zorder=15)
    ax_a.text(np.sqrt(X_MIN * X_MAX_A), e_eff_a - 0.18,
              'Model-bound',
              fontsize=FONT_REGION, ha='center', va='top',
              color=COLOR_MODEL_DARK, fontstyle='italic', fontweight='bold',
              zorder=15)

    ax_a.set_xscale('log')
    ax_a.set_xlim(X_MIN, X_MAX_A)
    ax_a.set_xticks([1e8, 1e9, 1e10])
    ax_a.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_a.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_a.set_ylim(FLOOR, L_max_a)
    ax_a.set_xlabel(r'Fresh tokens $D$',
                    fontsize=FONT_LABEL, fontweight='bold')
    ax_a.set_ylabel(r'Validation loss $L$',
                    fontsize=FONT_LABEL, fontweight='bold')
    ax_a.set_title('(a)', fontsize=FONT_TITLE, fontweight='bold')
    ax_a.tick_params(labelsize=FONT_TICK)
    ax_a.grid(True, alpha=0.2)

    # ---------- Panel (b): same size as (a), L vs D_eff ----------
    et_b = eta_of(D_a, Dp_a, N_a)
    D_eff_b = D_a + et_b * Dp_a

    X_MAX_B = 1.0e11
    xb = np.geomspace(X_MIN, X_MAX_B, 500)
    # Model-bound shading: below the irreducible loss.
    ax_b.fill_between([X_MIN, X_MAX_B], FLOOR, e_eff_a,
                      color=COLOR_MODEL_REGION, alpha=ALPHA_MODEL,
                      zorder=1)
    ax_b.plot(xb, L_chin(N_a, xb), '-', color=COLOR_FIT_1EP,
              linewidth=2.5, zorder=12)
    ax_b.axhline(e_eff_a, linestyle='-', color=COLOR_IRREDUCIBLE,
                 linewidth=2.5, zorder=12)
    ax_b.text(np.sqrt(X_MIN * X_MAX_B), e_eff_a - 0.18,
              'Model-bound', fontsize=FONT_REGION, ha='center', va='top',
              color=COLOR_MODEL_DARK, fontstyle='italic', fontweight='bold',
              zorder=15)

    ax_b.scatter(D_eff_b[~is_1ep_a], L_a[~is_1ep_a], s=S_MULTI,
                 c=log2F_a[~is_1ep_a], cmap=CMAP_COMPUTE, norm=flops_norm,
                 edgecolors='black', linewidths=0.7,
                 zorder=10, marker='o')
    ax_b.scatter(D_eff_b[is_1ep_a], L_a[is_1ep_a], s=S_1EP,
                 c=log2F_a[is_1ep_a], cmap=CMAP_COMPUTE, norm=flops_norm,
                 edgecolors='black', linewidths=0.9,
                 zorder=11, marker='o')

    ax_b.set_xscale('log')
    ax_b.set_xlim(X_MIN, X_MAX_B)
    ax_b.set_xticks([1e8, 1e9, 1e10, 1e11])
    ax_b.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_b.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_b.set_ylim(FLOOR, L_max_a)  # share y with panel (a)
    ax_b.set_xlabel(r"Effective tokens  $D_{\rm eff} = D + \eta\,D'$",
                    fontsize=FONT_LABEL, fontweight='bold')
    ax_b.set_ylabel('')
    plt.setp(ax_b.get_yticklabels(), visible=False)
    ax_b.set_title('(b)', fontsize=FONT_TITLE, fontweight='bold')
    ax_b.tick_params(labelsize=FONT_TICK)
    ax_b.grid(True, alpha=0.25)

    # ---------- Panel (c): D + D' vs D + eta D' ----------
    panel_c_sizes = [(t, N, m) for (t, N, m) in SIZES if t in PANEL_C_TAGS]

    # Smooth saturation curves: one curve per (N, observed scale). Colour by
    # N (plasma); each scale traces its own trajectory bounded by D*(1+R*).
    epochs_grid = np.geomspace(1.0, 256.0, 300)
    for tag, N, _ in panel_c_sizes:
        _, pts = points[tag]
        if len(pts) == 0:
            continue
        scales_obs = np.unique(pts[:, 0])
        for sc in scales_obs:
            D = sc * TTP_RATIO * N
            Dp = (epochs_grid - 1.0) * D
            et = eta_of(D, Dp, N)
            total = D + Dp
            D_eff = D + et * Dp
            ax_c.plot(total, D_eff, '-', color=colors[tag],
                      linewidth=2.0, alpha=0.45, zorder=4)
        # one labeled handle per N for the legend (use the 1× curve as proxy)
        D = 1.0 * TTP_RATIO * N
        Dp = (epochs_grid - 1.0) * D
        et = eta_of(D, Dp, N)
        ax_c.plot([], [], '-', color=colors[tag], linewidth=3.0, alpha=0.7,
                  label=tag)

    # Best-D_eff point per (N, scale): the run with the highest D_eff
    # across all epochs at that fresh-token level.  Marker size encodes
    # the chinchilla scale (D/N).
    def size_for_DN(D_over_N):
        # Map log2(D/N) ∈ [0, 9] to marker size in [15, 140].
        v = np.clip(np.log2(np.maximum(D_over_N, 1.0)), 0.0, 9.0)
        return 15.0 + 14.0 * v

    for tag, N, _ in panel_c_sizes:
        N_val, pts = points[tag]
        if len(pts) == 0:
            continue
        scale, ep, D, Dp, L, F = pts.T
        # Group by unique D (= unique scale at this N).
        for D_unique in np.unique(D):
            mask = (D == D_unique) & (ep > 1)
            if not mask.any():
                continue
            et = eta_of(D[mask], Dp[mask], N_val)
            D_eff = D[mask] + et * Dp[mask]
            j = np.argmax(D_eff)
            x_pt = D[mask][j] + Dp[mask][j]
            y_pt = D_eff[j]
            ax_c.scatter([x_pt], [y_pt],
                         s=size_for_DN(D_unique / N_val),
                         color=colors[tag], edgecolors='black',
                         linewidths=0.7, alpha=0.9, zorder=10)

    # identity line — drawn on top so it's visible over curves and points
    xlo, xhi = X_MIN, 4e11
    ylo, yhi = X_MIN, 1e11
    yy = np.geomspace(xlo, xhi, 100)
    ax_c.plot(yy, yy, '--', color='black', alpha=0.7, linewidth=1.8,
              zorder=15, label=r"$y = x$  (full credit)")

    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.set_xlim(xlo, xhi)
    ax_c.set_ylim(ylo, yhi)
    ax_c.set_xticks([1e8, 1e9, 1e10, 1e11])
    ax_c.set_yticks([1e8, 1e9, 1e10, 1e11])
    ax_c.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_c.yaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax_c.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_c.yaxis.set_minor_formatter(plt.NullFormatter())
    ax_c.set_xlabel(r"Total tokens trained  $D + D'$",
                    fontsize=FONT_LABEL, fontweight='bold')
    ax_c.set_ylabel(r"Effective fresh tokens  $D + \eta\,D'$",
                    fontsize=FONT_LABEL, fontweight='bold')
    ax_c.set_title('(c)', fontsize=FONT_TITLE, fontweight='bold')
    ax_c.tick_params(labelsize=FONT_TICK)
    ax_c.grid(True, which='both', alpha=0.25)
    leg_n = ax_c.legend(loc='lower right', fontsize=FONT_LEGEND, ncol=2,
                        columnspacing=1.0, handletextpad=0.4, title=r'$N$',
                        title_fontsize=FONT_LEGEND)
    ax_c.add_artist(leg_n)

    # Size legend: marker size encodes TTP (D/N).
    from matplotlib.lines import Line2D
    size_handles = []
    for ttp_val in (2, 20, 320):
        sz = np.sqrt(size_for_DN(ttp_val))  # markersize is sqrt of s
        size_handles.append(
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor='0.55', markeredgecolor='black',
                   markeredgewidth=0.5, markersize=sz,
                   label=rf'${ttp_val}$')
        )
    ax_c.legend(handles=size_handles, loc='upper left',
                fontsize=FONT_LEGEND, title='TTP (D/N)',
                title_fontsize=FONT_LEGEND, frameon=True,
                labelspacing=1.2, borderpad=0.8, handletextpad=0.6)

    out_pdf = os.path.join(os.path.dirname(__file__), 'figure_1.pdf')
    out_png = out_pdf.replace('.pdf', '.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf}\n      {out_png}')


if __name__ == '__main__':
    main()
