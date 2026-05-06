"""Paraphrase data-optimal scaling-law figures.

One figure per size (14M, 30M, 60M, 190M), each showing the same content
as data_optimal_scaling_iso.py but for the **paraphrase** η-surface
instead of repetition:

  * 1-epoch fit (black)
  * paraphrase D'/D -> inf fit (red)
  * irreducible loss (green)
  * FLOPs-coloured iso-D'/D dashed curves between them
  * paraphrase scatter points (squares) coloured by FLOPs
  * region labels (Compute-bound / Data-bound / Model-bound)
  * action arrows ("add paraphrases", "add fresh data")

Scaling-law anchors are the §6.2 triple joint fit (k=15) from
results/chinchilla_fit_dolma/writeup.md.

Saves one PDF + PNG per size to results/.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))


# === Scaling-law parameters ===
# fit_joint_triple_v2_nopen_14mpara.json — triple joint fit on 1-ep +
# repetition + paraphrase. N = non-embedding params; 14M dropped from
# 1-ep + rep but its paraphrase points kept; **no penalty, no
# weighting**. The extra 14M paraphrase points pin σ_para naturally
# negative. Canonical residual-drop k=20 (n_kept=290 / 310).
SL_E     = 1.1739
SL_A     = 212.04
SL_B     = 5542.28
SL_ALPHA = 0.2913
SL_BETA  = 0.3727
# η_para surface
SL_LOG_K_PARA = 36.8210
SL_RHO_PARA   = -2.1965
SL_SIGMA_PARA = -1.5060


def E_eff(N):
    return SL_E + SL_A / (N ** SL_ALPHA)


def L_1ep(D, N):
    D = np.asarray(D, dtype=float)
    return E_eff(N) + SL_B / D ** SL_BETA


def R_star_para(D, N):
    D = np.asarray(D, dtype=float)
    return np.exp(SL_LOG_K_PARA
                  + SL_RHO_PARA * np.log(D / N)
                  + SL_SIGMA_PARA * np.log(N))


def L_inf_para(D, N):
    """Paraphrase limit D' -> infinity."""
    D = np.asarray(D, dtype=float)
    D_eff = D * (1.0 + R_star_para(D, N))
    return E_eff(N) + SL_B / D_eff ** SL_BETA


def L_iso_x_para(D, x, N):
    """Loss at fixed D'/D = x for the paraphrase η-surface."""
    D = np.asarray(D, dtype=float)
    if x == 0:
        return L_1ep(D, N)
    R = R_star_para(D, N)
    eta_times_x = R * (1.0 - np.exp(-x / R))
    return E_eff(N) + SL_B / (D * (1.0 + eta_times_x)) ** SL_BETA


# === Visual constants (match data_optimal_scaling_iso.py) ===
FONT_LABEL  = 26
FONT_LEGEND = 18
FONT_TICK   = 22
FONT_ANNOT  = 22
FONT_REGION = 22
FONT_TITLE  = 28
FONT_CBAR_LABEL = 24
FONT_CBAR_TICK  = 20

CMAP = LinearSegmentedColormap.from_list(
    'compute',
    ['#A8E6CF', '#5BC4B0', '#3B9AB2', '#2C6E91', '#1F4E79', '#152F4F', '#0A1628'],
)

COLOR_COMPUTE_REGION = '#A8C5DC'
COLOR_DATA_REGION    = '#E8B4A0'
COLOR_MODEL_REGION   = '#C8D5B9'
ALPHA_COMPUTE = 0.35
ALPHA_DATA    = 0.30
ALPHA_MODEL   = 0.35

COLOR_COMPUTE_DARK = '#1F4E79'
COLOR_DATA_DARK    = '#8B4513'
COLOR_MODEL_DARK   = '#1b5e20'

COLOR_ARROW          = '#555555'
COLOR_TEXT           = '#333333'
COLOR_FIT_1EP        = 'black'
COLOR_FIT_INF        = '#cb181d'
COLOR_IRREDUCIBLE    = '#1b5e20'
COLOR_DATA_HIGHLIGHT = '#8B0000'

ISO_X_VALUES = [1, 3, 7, 15, 31, 63]


def _collect_para(parap_datasets, N):
    """Return (D, L, flops, x=D'/D) arrays for paraphrase scatter points."""
    D_arr, L_arr, F_arr, X_arr = [], [], [], []
    for pds in parap_datasets:
        scale = pds['chinchilla_scale'][0]
        D = scale * 20.0 * N
        for tt, fi, li in zip(pds['tokens_trained'],
                              pds['flops_multiplier'],
                              pds['validation_loss']):
            if li is None or np.isnan(li):
                continue
            Dp = float(tt) - D
            if Dp <= 0:
                continue
            D_arr.append(D)
            L_arr.append(li)
            F_arr.append(fi)
            X_arr.append(Dp / D)
    return (np.array(D_arr), np.array(L_arr),
            np.array(F_arr), np.array(X_arr))


def _fmt_d(v):
    if v >= 1e9:
        return f'{v/1e9:g}B'
    if v >= 1e6:
        return f'{v/1e6:g}M'
    return f'{v:g}'


# N = non-embedding params (matches fit_joint_triple_v2 fit).
# 14M is plotted as a held-out check — it was excluded from the fit.
SIZES = [
    ('dolma_14m',  13_895_808),
    ('dolma_30m',  29_102_336),
    ('dolma_60m',  57_422_208),
    ('dolma_190m', 190_354_176),
]


def joint_flops_norm():
    """Joint FLOPs normalisation across all paraphrase points in all sizes."""
    all_f = []
    for ms, N in SIZES:
        md = __import__(ms)
        if not getattr(md, 'parap_datasets', None):
            continue
        _, _, f, _ = _collect_para(md.parap_datasets, N)
        if f.size:
            all_f.append(f)
    af = np.concatenate(all_f)
    return plt.Normalize(vmin=np.log2(af.min()), vmax=np.log2(af.max()))


def _render_panel(ax, model_size, N, norm):
    md = __import__(model_size)
    parap = getattr(md, 'parap_datasets', None) or []
    D_pts, L_pts, F_pts, _ = _collect_para(parap, N)

    sc = ax.scatter(D_pts, L_pts,
                    c=np.log2(F_pts), cmap=CMAP, norm=norm,
                    marker='s', s=85, edgecolors='black', linewidths=1.0,
                    zorder=10, label='Paraphrase runs')

    XLEFT  = D_pts.min() * 0.5
    XRIGHT = max(D_pts.max() * 2.0, 14e6 * 100)
    e_eff  = E_eff(N)
    FLOOR  = e_eff - 0.45
    Y_TOP  = max(L_pts.max() + 0.6, 9.0)

    xi      = np.geomspace(XLEFT, XRIGHT, 500)
    l1ep_xi = L_1ep(xi, N)
    linf_xi = L_inf_para(xi, N)

    # Region fills.
    ax.fill_between(xi, linf_xi, l1ep_xi, where=l1ep_xi > linf_xi,
                    color=COLOR_COMPUTE_REGION, alpha=ALPHA_COMPUTE, zorder=2,
                    label='Compute-bound')
    ax.fill_between(xi, e_eff, linf_xi, where=linf_xi > e_eff,
                    color=COLOR_DATA_REGION, alpha=ALPHA_DATA, zorder=3,
                    label='Data-bound')
    ax.fill_between([XLEFT, XRIGHT], FLOOR, e_eff,
                    color=COLOR_MODEL_REGION, alpha=ALPHA_MODEL, zorder=1,
                    label='Model-bound')

    # 1-epoch fit + iso-D'/D dashed curves coloured by FLOPs + paraphrase ∞ fit.
    ax.plot(xi, l1ep_xi, linestyle='-', color=COLOR_FIT_1EP,
            linewidth=3.0, alpha=0.95, zorder=12,
            label=r"CD-law: $D'/D = 0$")
    for x in ISO_X_VALUES:
        L_xi = L_iso_x_para(xi, x, N)
        flops_xi = (xi / (20 * N)) * (1.0 + x)
        log2_flops = np.log2(flops_xi)
        pts = np.array([xi, L_xi]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=CMAP, norm=norm,
                            linestyles='dashed', linewidths=1.6,
                            alpha=0.95, zorder=11)
        lc.set_array(0.5 * (log2_flops[:-1] + log2_flops[1:]))
        ax.add_collection(lc)
    ax.plot(xi, linf_xi, linestyle='-', color=COLOR_FIT_INF,
            linewidth=3.0, alpha=0.95, zorder=12,
            label=r"CD-law: $D'/D \to \infty$ (paraphrase)")
    ax.axhline(e_eff, linestyle='-', color=COLOR_IRREDUCIBLE,
               linewidth=3.0, alpha=0.95, zorder=12,
               label=r"CD-law: $D' \to \infty,\ D \to \infty$")

    ax.set_xlabel('Fresh Data Size,  D (tokens)',
                  fontsize=FONT_LABEL, fontweight='bold', labelpad=12)
    ax.set_ylabel('Validation Loss',
                  fontsize=FONT_LABEL, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xlim(XLEFT, XRIGHT)
    ax.set_ylim(FLOOR, Y_TOP)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: _fmt_d(v)))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # === In-region text labels ===
    x_mid_log = np.sqrt(XLEFT * XRIGHT)

    # Compute-bound: shift right so it doesn't collide with the arrow.
    x_cb_text = np.sqrt(x_mid_log * XRIGHT)
    y_cb_text = float(L_1ep(x_cb_text, N)) + 0.55
    if y_cb_text < Y_TOP - 0.2:
        ax.text(x_cb_text, y_cb_text, 'Compute-bound',
                fontsize=FONT_REGION, ha='center', va='bottom',
                color=COLOR_COMPUTE_DARK, fontstyle='italic',
                fontweight='bold', zorder=15)

    x_db = np.sqrt(x_mid_log * XRIGHT)
    y_db = (float(L_inf_para(x_db, N)) + e_eff) / 2
    ax.text(x_db, y_db, 'Data-bound',
            fontsize=FONT_REGION, ha='center', va='center',
            color=COLOR_DATA_DARK, fontstyle='italic', fontweight='bold',
            zorder=15)

    ax.text(x_mid_log, e_eff - 0.18,
            'Model-bound  (need larger N)',
            fontsize=FONT_REGION, ha='center', va='top',
            color=COLOR_MODEL_DARK, fontstyle='italic', fontweight='bold',
            zorder=15)

    # === Action arrows ===
    arrow_x = 0.25 * 20 * N
    if arrow_x > XLEFT and arrow_x < XRIGHT:
        y_top = float(L_1ep(arrow_x, N)) - 0.15
        y_bot = float(L_inf_para(arrow_x, N)) + 0.20
        if y_top > y_bot:
            ax.annotate('', xy=(arrow_x, y_bot), xytext=(arrow_x, y_top),
                        arrowprops=dict(arrowstyle='-|>', color=COLOR_ARROW,
                                        lw=3.0, mutation_scale=28),
                        zorder=15)
            ax.text(arrow_x * 1.20, (y_top + y_bot) / 2,
                    'add paraphrases\n(more K)',
                    fontsize=FONT_ANNOT, ha='left', va='center',
                    color=COLOR_TEXT, fontweight='bold', zorder=15)

    x_a, x_b = 0.08 * 20 * N, 0.5 * 20 * N
    if x_a > XLEFT and x_b < XRIGHT:
        y_a, y_b = float(L_inf_para(x_a, N)), float(L_inf_para(x_b, N))
        ax.annotate('', xy=(x_b, y_b), xytext=(x_a, y_a),
                    arrowprops=dict(arrowstyle='-|>',
                                    color=COLOR_DATA_HIGHLIGHT,
                                    lw=3.0, mutation_scale=28),
                    zorder=15)
        # Place label above the arrow rather than below, since paraphrase
        # R* is large enough that the D'/D->inf curve sits close to e_eff.
        ax.text(np.sqrt(x_a * x_b), max(y_a, y_b) + 0.25,
                'add fresh data',
                fontsize=FONT_ANNOT, ha='center', va='bottom',
                color=COLOR_DATA_HIGHLIGHT, fontweight='bold', zorder=15)

    return sc


def plot_one(model_size, N, norm):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(left=0.10, right=0.92, top=0.93, bottom=0.22)
    sc = _render_panel(ax, model_size, N, norm)

    ax.set_title(f'{N/1e6:.1f}M non-emb (paraphrase)',
                 fontsize=FONT_TITLE, fontweight='bold', pad=10)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('FLOPs (Chinchilla Optimal = 1X)',
                   fontsize=FONT_CBAR_LABEL, fontweight='bold')
    cbar.ax.tick_params(labelsize=FONT_CBAR_TICK)
    flops_ticks = [0.05, 0.5, 1, 8, 64, 1024]
    cbar.set_ticks([np.log2(v) for v in flops_ticks])
    cbar.set_ticklabels([f'{v:g}' for v in flops_ticks])

    handles, labels = ax.get_legend_handles_labels()
    handles_map = dict(zip(labels, handles))

    section_handle = Line2D([0], [0], color='none', linewidth=0)
    handles_map['CD-Scaling Law Prediction'] = section_handle

    all_keys = ['CD-Scaling Law Prediction',
                r"CD-law: $D'/D = 0$",
                r"CD-law: $D'/D \to \infty$ (paraphrase)",
                r"CD-law: $D' \to \infty,\ D \to \infty$",
                'Paraphrase runs',
                'Compute-bound', 'Data-bound', 'Model-bound']
    leg_handles = [handles_map[k] for k in all_keys if k in handles_map]
    leg_labels  = [k for k in all_keys if k in handles_map]
    leg = fig.legend(leg_handles, leg_labels,
                     loc='lower center', bbox_to_anchor=(0.5, -0.12),
                     ncol=4, frameon=False, fontsize=FONT_LEGEND)
    if 'CD-Scaling Law Prediction' in leg_labels:
        idx = leg_labels.index('CD-Scaling Law Prediction')
        leg.get_texts()[idx].set_color(COLOR_COMPUTE_DARK)
        leg.get_texts()[idx].set_fontweight('bold')

    out_pdf = f'results/data_optimal_scaling_iso_para_{model_size}.pdf'
    out_png = out_pdf.replace('.pdf', '.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight', dpi=140)
    plt.close(fig)
    print(f'Saved {out_pdf} and {out_png}')


if __name__ == '__main__':
    norm = joint_flops_norm()
    for ms, N in SIZES:
        plot_one(ms, N, norm)
