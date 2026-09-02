"""Combined data-optimal scaling-law figure.

Single figure with:
  * Left half — full panel for one model size (default 30M):
      1-epoch fit (black), multi-epoch D'/D -> inf fit (red), irreducible
      loss (green), and FLOPs-coloured iso-D'/D dashed curves between
      them. In-region text labels for Compute-bound / Data-bound /
      Model-bound. Annotation arrows for "add compute" and "add fresh
      data" (dark crimson).
  * Right half — 2x2 small multiples (30M, 60M, 190M, 370M) in TTP
      space (D/N) so all four panels share a common x-range, showing
      how the data-bound and compute-bound bands shrink with N.

Shared across all five panels:
  * Single FLOPs colorbar on the right.
  * Single legend at the bottom of the figure.

Region semantics (corrected):
  * between 1-epoch and multi-epoch fits      -> Compute-bound
  * between multi-epoch fit and irreducible   -> Data-bound
  * below irreducible (unreachable for this N) -> Model-bound

Scaling-law anchors are the canonical one-shot joint fit from
results/chinchilla_fit_dolma/writeup_final.md (k=15, all 7 sizes).
"""
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))


# Palatino Linotype (bundled in results/fonts/) — matches isoloss_contour styling.
_FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
for _f in glob.glob(os.path.join(_FONT_DIR, 'palatinolinotype_*.ttf')):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Palatino Linotype', 'P052', 'Palatino', 'serif'],
    'mathtext.fontset': 'cm',
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


def E_eff(N):
    return SL_E + SL_A / (N ** SL_ALPHA)


def L_1ep(D, N):
    D = np.asarray(D, dtype=float)
    return E_eff(N) + SL_B / D ** SL_BETA


def R_star(D, N):
    D = np.asarray(D, dtype=float)
    return np.exp(SL_LOG_K + SL_RHO * np.log(D / N) + SL_SIGMA * np.log(N))


def L_inf_epochs(D, N):
    """Multi-epoch limit D' -> infinity (exp-sat R*(N))."""
    D = np.asarray(D, dtype=float)
    D_eff = D * (1.0 + R_star(D, N))
    return E_eff(N) + SL_B / D_eff ** SL_BETA


def L_iso_x(D, x, N):
    """Loss at fixed D'/D = x (epochs = 1 + x)."""
    D = np.asarray(D, dtype=float)
    if x == 0:
        return L_1ep(D, N)
    R = R_star(D, N)
    eta_times_x = R * (1.0 - np.exp(-x / R))
    return E_eff(N) + SL_B / (D * (1.0 + eta_times_x)) ** SL_BETA


# === Visual constants ===
# Match the typography of results/isoloss_contour.py so both figures look
# consistent in the same paper.
FONT_LABEL  = 30
FONT_LEGEND = 24
FONT_TICK   = 26
FONT_ANNOT  = 28
FONT_REGION = 28
FONT_TITLE  = 30
FONT_CBAR_LABEL = 26
FONT_CBAR_TICK  = 22

CMAP = LinearSegmentedColormap.from_list(
    'compute',
    ['#A8E6CF', '#5BC4B0', '#3B9AB2', '#2C6E91', '#1F4E79', '#152F4F', '#0A1628'],
)

COLOR_COMPUTE_REGION = '#A8C5DC'   # soft blue       — compute-bound
COLOR_DATA_REGION    = '#E8B4A0'   # muted terracotta — data-bound
COLOR_MODEL_REGION   = '#C8D5B9'   # sage             — model-bound (unreachable)
ALPHA_COMPUTE = 0.35
ALPHA_DATA    = 0.30
ALPHA_MODEL   = 0.35

COLOR_COMPUTE_DARK   = '#1F4E79'   # dark navy (compute-bound text)
COLOR_DATA_DARK      = '#8B4513'   # dark terracotta (data-bound text)
COLOR_MODEL_DARK     = '#1b5e20'   # dark green (model-bound text)

COLOR_ARROW          = '#555555'
COLOR_TEXT           = '#333333'
COLOR_FIT_1EP        = 'black'
COLOR_FIT_MEINF      = '#cb181d'   # red — multi-epoch fit curve
COLOR_IRREDUCIBLE    = '#1b5e20'   # dark green — irreducible loss
COLOR_DATA_HIGHLIGHT = '#8B0000'   # dark crimson — "add fresh data"

ISO_X_VALUES = [1, 3, 7, 15, 31, 63]


def _collect(datasets):
    c, l, f = [], [], []
    for d in datasets:
        chin = d['chinchilla_scale'][0]
        for fi, li in zip(d['flops_multiplier'], d['validation_loss']):
            if not np.isnan(li):
                c.append(chin)
                l.append(li)
                f.append(fi)
    return np.array(c), np.array(l), np.array(f)


def _fmt_d(v):
    if v >= 1e9:
        return f'{v/1e9:g}B'
    if v >= 1e6:
        return f'{v/1e6:g}M'
    return f'{v:g}'


def actual_flops(flops_multiplier, N):
    """FLOPs = 6·N·D_total = 120·N²·flops_multiplier (since C* = 120 N²)."""
    return np.asarray(flops_multiplier) * 120.0 * N ** 2


def joint_flops_norm(sizes):
    """Norm over log2(actual FLOPs) across all panels' data."""
    all_f = []
    for model_size, N in sizes:
        md = __import__(model_size)
        _, _, f = _collect(md.ALL_DATASETS)
        all_f.append(actual_flops(f, N))
    af = np.concatenate(all_f)
    return plt.Normalize(vmin=np.log2(af.min()), vmax=np.log2(af.max()))


# ---- Figure A panel renderer ------------------------------------------------

def _render_panel_a(ax, model_size, N, norm):
    md = __import__(model_size)
    chin, loss, flops = _collect(md.ALL_DATASETS)

    sc = ax.scatter(chin * 20.0, loss,
                    c=np.log2(actual_flops(flops, N)), cmap=CMAP, norm=norm,
                    s=85, edgecolors='black', linewidths=1.0, zorder=10,
                    label='Models trained')

    # X axis is TTP (= D/N) so it lines up with the small-multiples panels.
    XLEFT_TTP  = min(chin) * 20.0 * 0.5
    XRIGHT_TTP = 500.0
    e_eff  = E_eff(N)
    FLOOR  = e_eff - 0.45
    Y_TOP  = max(loss.max() + 0.6, 9.0)

    ttp_grid = np.geomspace(XLEFT_TTP, XRIGHT_TTP, 500)
    D_grid   = ttp_grid * N
    l1ep_xi  = L_1ep(D_grid, N)
    lme_xi   = L_inf_epochs(D_grid, N)

    # Region fills.
    ax.fill_between(ttp_grid, lme_xi, l1ep_xi, where=l1ep_xi > lme_xi,
                    color=COLOR_COMPUTE_REGION, alpha=ALPHA_COMPUTE, zorder=2,
                    label='Compute-bound')
    ax.fill_between(ttp_grid, e_eff, lme_xi, where=lme_xi > e_eff,
                    color=COLOR_DATA_REGION, alpha=ALPHA_DATA, zorder=3,
                    label='Data-bound')
    ax.fill_between([XLEFT_TTP, XRIGHT_TTP], FLOOR, e_eff,
                    color=COLOR_MODEL_REGION, alpha=ALPHA_MODEL, zorder=1,
                    label='Model-bound')

    # 1-epoch fit + iso-D'/D dashed curves coloured by FLOPs + multi-epoch fit.
    ax.plot(ttp_grid, l1ep_xi, linestyle='-', color=COLOR_FIT_1EP,
            linewidth=3.0, alpha=0.95, zorder=12,
            label=r"CD-law: $D'/D = 0$")
    for x in ISO_X_VALUES:
        L_xi = L_iso_x(D_grid, x, N)
        flops_mult_xi = (ttp_grid / 20.0) * (1.0 + x)
        log2_flops = np.log2(actual_flops(flops_mult_xi, N))
        pts = np.array([ttp_grid, L_xi]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=CMAP, norm=norm,
                            linestyles='dashed', linewidths=1.6,
                            alpha=0.95, zorder=11)
        lc.set_array(0.5 * (log2_flops[:-1] + log2_flops[1:]))
        ax.add_collection(lc)
    ax.plot(ttp_grid, lme_xi, linestyle='-', color=COLOR_FIT_MEINF,
            linewidth=3.0, alpha=0.95, zorder=12,
            label=r"CD-law: $D'/D \to \infty$")
    ax.axhline(e_eff, linestyle='-', color=COLOR_IRREDUCIBLE,
               linewidth=3.0, alpha=0.95, zorder=12,
               label=r"CD-law: $D \to \infty$")

    ax.set_xlabel('TPP (D/N)',
                  fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Validation Loss',
                  fontsize=FONT_LABEL, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xlim(XLEFT_TTP, XRIGHT_TTP)
    ax.set_ylim(FLOOR, Y_TOP)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_xticks([1, 10, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # === In-region text labels ===
    # Compute-bound: blue guideline arrow from the text down into the
    # compute-bound (blue) band, mirroring the "add fresh data" treatment.
    ttp_cb_text = 80.0
    y_cb_text   = float(L_1ep(ttp_cb_text * N, N)) + 1.10
    ttp_cb_tip  = 20.0
    y_cb_tip    = 0.5 * (float(L_1ep(ttp_cb_tip * N, N))
                         + float(L_inf_epochs(ttp_cb_tip * N, N)))
    ax.annotate('Compute-bound',
                xy=(ttp_cb_tip, y_cb_tip),
                xytext=(ttp_cb_text, y_cb_text),
                fontsize=FONT_REGION, ha='center', va='bottom',
                color=COLOR_COMPUTE_DARK, fontstyle='italic', fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color=COLOR_COMPUTE_DARK,
                                lw=2.5, mutation_scale=24,
                                shrinkA=4, shrinkB=2,
                                connectionstyle='arc3,rad=0.25'),
                zorder=15)

    # Data-bound: in the sage band, where it's still wide (low/mid D).
    ttp_db = 33.0
    y_db = (float(L_inf_epochs(ttp_db * N, N)) + e_eff) / 2
    ax.text(ttp_db, y_db, 'Data-bound',
            fontsize=FONT_REGION, ha='center', va='center',
            color=COLOR_DATA_DARK, fontstyle='italic', fontweight='bold',
            zorder=15)

    # Model-bound: below the irreducible-loss line.
    ax.text(np.sqrt(XLEFT_TTP * XRIGHT_TTP), e_eff - 0.18,
            'Model-bound',
            fontsize=FONT_REGION, ha='center', va='top',
            color=COLOR_MODEL_DARK, fontstyle='italic', fontweight='bold',
            zorder=15)

    # === Action arrows ===
    # "add compute": vertical arrow inside the compute-bound region. Shifted
    # to the left so the arrowhead/text don't overlap the data scatter cluster.
    arrow_ttp = 2.5
    y_top = float(L_1ep(arrow_ttp * N, N)) - 0.15
    y_bot = float(L_inf_epochs(arrow_ttp * N, N)) + 0.20
    ax.annotate('', xy=(arrow_ttp, y_bot), xytext=(arrow_ttp, y_top),
                arrowprops=dict(arrowstyle='-|>', color=COLOR_ARROW, lw=3.0,
                                mutation_scale=28),
                zorder=15)
    ax.text(arrow_ttp * 1.20, (y_top + y_bot) / 2,
            'add compute',
            fontsize=FONT_ANNOT, ha='left', va='center',
            color=COLOR_TEXT, fontweight='bold', zorder=15)

    # "add fresh data": red guideline arrow from the text up to a point on
    # the multi-epoch (red) curve, so it's visually anchored to that line.
    ttp_text = 1.0
    y_text   = float(L_inf_epochs(ttp_text * N, N)) - 1.10
    ttp_tip  = 5.0
    y_tip    = float(L_inf_epochs(ttp_tip * N, N))
    ax.annotate('add fresh data',
                xy=(ttp_tip, y_tip),
                xytext=(ttp_text, y_text),
                fontsize=FONT_ANNOT, ha='left', va='top',
                color=COLOR_DATA_HIGHLIGHT, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color=COLOR_DATA_HIGHLIGHT,
                                lw=2.5, mutation_scale=24,
                                shrinkA=4, shrinkB=2,
                                connectionstyle='arc3,rad=-0.25'),
                zorder=15)

    return sc


# ---- Figure B panel renderer (TTP space) ------------------------------------

def _render_panel_b(ax, model_size, N, norm, ttp_range, y_range):
    md = __import__(model_size)
    chin, loss, flops = _collect(md.ALL_DATASETS)

    XLEFT_TTP, XRIGHT_TTP = ttp_range
    floor, y_top = y_range
    e_eff = E_eff(N)

    sc = ax.scatter(chin * 20.0, loss,
                    c=np.log2(actual_flops(flops, N)), cmap=CMAP, norm=norm,
                    s=45, edgecolors='black', linewidths=0.6, zorder=10,
                    label='Models trained')

    ttp_grid = np.geomspace(XLEFT_TTP, XRIGHT_TTP, 500)
    D_grid   = ttp_grid * N
    l1ep_xi  = L_1ep(D_grid, N)
    lme_xi   = L_inf_epochs(D_grid, N)

    ax.fill_between(ttp_grid, lme_xi, l1ep_xi, where=l1ep_xi > lme_xi,
                    color=COLOR_COMPUTE_REGION, alpha=ALPHA_COMPUTE, zorder=2,
                    label='Compute-bound')
    ax.fill_between(ttp_grid, e_eff, lme_xi, where=lme_xi > e_eff,
                    color=COLOR_DATA_REGION, alpha=ALPHA_DATA, zorder=3,
                    label='Data-bound')
    ax.fill_between([XLEFT_TTP, XRIGHT_TTP], floor, e_eff,
                    color=COLOR_MODEL_REGION, alpha=ALPHA_MODEL, zorder=1,
                    label='Model-bound')

    ax.plot(ttp_grid, l1ep_xi, linestyle='-', color=COLOR_FIT_1EP,
            linewidth=2.3, alpha=0.95, zorder=12,
            label=r"CD-law: $D'/D = 0$")
    ax.plot(ttp_grid, lme_xi, linestyle='-', color=COLOR_FIT_MEINF,
            linewidth=2.3, alpha=0.95, zorder=12,
            label=r"CD-law: $D'/D \to \infty$")
    ax.axhline(e_eff, linestyle='-', color=COLOR_IRREDUCIBLE,
               linewidth=2.3, alpha=0.95, zorder=12,
               label=r"CD-law: $D \to \infty$")

    ax.set_xscale('log', base=10)
    ax.set_xlim(XLEFT_TTP, XRIGHT_TTP)
    ax.set_ylim(floor, y_top)
    ax.set_xticks([1, 10, 100])
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_title(f'{int(N/1e6)}M', fontsize=FONT_TITLE, fontweight='bold',
                 pad=8)

    return sc


# ---- Combined figure --------------------------------------------------------

SIZES_B = [
    ('dolma_30m',  30e6),
    ('dolma_60m',  60e6),
    ('dolma_190m', 190e6),
    ('dolma_370m', 370e6),
]


def plot_combined():
    norm = joint_flops_norm(SIZES_B)

    # Shared y-range for all Figure B panels.
    y_top = 0.0
    floor = 1e9
    for ms, N in SIZES_B:
        md = __import__(ms)
        _, loss, _ = _collect(md.ALL_DATASETS)
        y_top = max(y_top, loss.max() + 0.6, 9.0)
        floor = min(floor, E_eff(N) - 0.45)
    ttp_range = (0.5, 500.0)
    y_range = (floor, y_top)

    fig, axes = plt.subplot_mosaic(
        [['A', 'A', 'B30', 'B60', 'cb'],
         ['A', 'A', 'B190', 'B370', 'cb']],
        figsize=(22, 11),
        gridspec_kw={'width_ratios': [1.0, 1.0, 0.7, 0.7, 0.07],
                     'wspace': 0.22, 'hspace': 0.25,
                     'left': 0.05, 'right': 0.93,
                     'top': 0.96, 'bottom': 0.22},
    )

    # Render Figure A on the left (default 30M).
    sc = _render_panel_a(axes['A'], 'dolma_30m', 30e6, norm)

    # Render Figure B 2x2 on the right.
    panel_keys = ['B30', 'B60', 'B190', 'B370']
    b_axes = [axes[k] for k in panel_keys]
    for ax, (ms, N) in zip(b_axes, SIZES_B):
        _render_panel_b(ax, ms, N, norm, ttp_range, y_range)

    # Share x/y across the four Figure B panels.
    for ax in b_axes[1:]:
        ax.sharex(b_axes[0])
        ax.sharey(b_axes[0])
    plt.setp(axes['B30'].get_xticklabels(), visible=False)
    plt.setp(axes['B60'].get_xticklabels(), visible=False)
    plt.setp(axes['B60'].get_yticklabels(), visible=False)
    plt.setp(axes['B370'].get_yticklabels(), visible=False)

    axes['B190'].set_xlabel('TPP', fontsize=FONT_LABEL, fontweight='bold')
    axes['B370'].set_xlabel('TPP', fontsize=FONT_LABEL, fontweight='bold')

    # Single shared FLOPs colorbar on the right (actual FLOPs = 6·N·D_total).
    # Pick decade ticks that fall inside the joint norm range so the bar fills.
    cbar = fig.colorbar(sc, cax=axes['cb'])
    cbar.set_label('FLOPs',
                   fontsize=FONT_CBAR_LABEL, fontweight='bold')
    cbar.ax.tick_params(labelsize=FONT_CBAR_TICK)
    decade_min = int(np.ceil(norm.vmin / np.log2(10)))
    decade_max = int(np.floor(norm.vmax / np.log2(10)))
    decades = list(range(decade_min, decade_max + 1))
    cbar.set_ticks([d * np.log2(10) for d in decades])
    cbar.set_ticklabels([rf'$10^{{{d}}}$' for d in decades])

    # === Single combined legend at the bottom of the figure ===
    handles, labels = axes['A'].get_legend_handles_labels()
    handles_map = dict(zip(labels, handles))

    # Section header rendered as a regular legend entry: invisible swatch
    # plus a blue, bold text label.
    section_handle = Line2D([0], [0], color='none', linewidth=0)
    handles_map['CD-Scaling Law Prediction'] = section_handle

    all_keys = ['CD-Scaling Law Prediction',
                r"CD-law: $D'/D = 0$",
                r"CD-law: $D'/D \to \infty$",
                r"CD-law: $D \to \infty$",
                'Models trained',
                'Compute-bound', 'Data-bound', 'Model-bound']
    leg = fig.legend([handles_map[k] for k in all_keys], all_keys,
                     loc='lower center', bbox_to_anchor=(0.5, 0.02),
                     ncol=4, frameon=False,
                     fontsize=FONT_LEGEND)
    # Style the "CD-Scaling Law Prediction" entry as a blue, bold header.
    leg.get_texts()[0].set_color(COLOR_COMPUTE_DARK)
    leg.get_texts()[0].set_fontweight('bold')

    out_pdf = 'results/data_optimal_scaling_iso_combined.pdf'
    out_png = out_pdf.replace('.pdf', '.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight', dpi=140)
    plt.close(fig)
    print(f'Saved {out_pdf} and {out_png}')


if __name__ == '__main__':
    plot_combined()
