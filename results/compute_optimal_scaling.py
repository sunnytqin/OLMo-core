import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# ---- Select model size ----
MODEL_SIZE = 'dolma_30m'

if MODEL_SIZE == '30m':
    import dclm_30m as model_data
elif MODEL_SIZE == '370m':
    import data_370m as model_data
elif MODEL_SIZE == 'dolma_14m':
    import dolma_14m as model_data
elif MODEL_SIZE == 'dolma_30m':
    import dolma_30m as model_data
elif MODEL_SIZE == 'dolma_60m':
    import dolma_60m as model_data
elif MODEL_SIZE == 'dolma_100m':
    import dolma_100m as model_data
elif MODEL_SIZE == 'dolma_190m':
    import dolma_190m as model_data
elif MODEL_SIZE == 'dolma_370m':
    import dolma_370m as model_data
elif MODEL_SIZE == 'dolma_600m':
    import dolma_600m as model_data
else:
    raise ValueError(f'Unknown MODEL_SIZE: {MODEL_SIZE}')

all_datasets = model_data.ALL_DATASETS
parap_datasets = getattr(model_data, 'parap_datasets', None)

FONT_LABEL  = 18
FONT_TITLE  = 20
FONT_LEGEND = 15
FONT_TICK   = 16

_ttp_labels = {
    0.05: '$D = 1N$ (0.05x Chin)',
    0.1:  '$D = 2N$ (0.1x Chin)',
    0.25: '$D = 5N$ (0.25x Chin)',
    0.5:  '$D = 10N$ (0.5x Chin)',
    1:    '$D = 20N$ (1x Chin)',
    2:    '$D = 40N$ (2x Chin)',
    4:    '$D = 80N$ (4x Chin)',
    8:    '$D = 160N$ (8x Chin)',
    16:   '$D = 320N$ (16x Chin)',
}

# Shared colormap across both panels (color = chin scale)
norm = plt.Normalize(vmin=-1, vmax=16)
cmap = plt.cm.magma_r


def compute_one_epoch_baseline(datasets):
    """Return (flops, loss) lists for the 1-epoch points across all chin scales."""
    xs, ys = [], []
    for d in datasets:
        epochs = np.array(d['epochs'])
        flops = np.array(d['flops_multiplier'], dtype=float)
        loss = np.array(d['validation_loss'], dtype=float)
        idx = np.where(epochs == 1)[0]
        if len(idx) > 0 and not np.isnan(loss[idx[0]]):
            xs.append(flops[idx[0]]); ys.append(loss[idx[0]])
    order = np.argsort(xs)
    return [xs[i] for i in order], [ys[i] for i in order]


def plot_panel(ax, datasets, baseline_xs, baseline_ys, title, axis_field='epochs'):
    """Render one compute-optimal panel.

    `datasets` are colored lines per chin scale (multi-epoch or paraphrase).
    `baseline_xs/ys` is the 1-epoch chinchilla line (same on both panels).
    `axis_field` is 'epochs' (multi-epoch) or 'K' (paraphrase) — used to skip empty entries.
    """
    # Baseline (dashed black) — drawn first so colored lines sit on top.
    compute_line, = ax.plot(baseline_xs, baseline_ys,
                            linestyle='--', color='black',
                            linewidth=7, alpha=0.7,
                            label='Compute Optimal (Chinchilla scaling)',
                            zorder=2)

    # Star at chinchilla optimal (chin=1, baseline) — looked up from baseline points.
    if 1.0 in baseline_xs:
        chin_star_loss = baseline_ys[baseline_xs.index(1.0)]
        ax.plot(1.0, chin_star_loss, marker='*', color='black',
                markersize=20, zorder=20, linestyle='none')
        ax.annotate('Chinchilla Optimal',
                    xy=(1.0, chin_star_loss),
                    xytext=(1.5, chin_star_loss + 0.05),
                    fontsize=FONT_LEGEND, color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    colored_handles, colored_labels = [], []
    for idx, data in enumerate(datasets):
        flops = np.array(data['flops_multiplier'], dtype=float)
        loss = np.array(data['validation_loss'], dtype=float)
        chin = data['chinchilla_scale'][0]
        valid = ~np.isnan(loss)
        if not valid.any():
            continue
        # Sort by flops so the line is monotonic on log-x
        order = np.argsort(flops[valid])
        x = flops[valid][order]; y = loss[valid][order]
        color = cmap(norm(chin))
        label = _ttp_labels.get(chin, f"{chin}x Chin")
        line, = ax.plot(x, y, linestyle='-', marker='o', color=color,
                        label=label, linewidth=2, markersize=5, zorder=10 + idx)
        colored_handles.append(line); colored_labels.append(label)

    ax.set_xlabel('FLOPs (Chinchilla Optimal = 1X)', fontsize=FONT_LABEL)
    ax.set_ylabel('Validation Loss', fontsize=FONT_LABEL)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:g}'))
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=FONT_TITLE)

    legend1 = ax.legend(colored_handles, colored_labels,
                        fontsize=FONT_LEGEND,
                        title='Fresh Data Size $D$',
                        title_fontsize=FONT_LEGEND,
                        loc='upper right')
    ax.add_artist(legend1)
    ax.legend(handles=[compute_line], labels=['Compute Optimal (Chinchilla)'],
              fontsize=FONT_LEGEND, loc='lower left')


# Baseline shared by both panels (1-epoch from multi-epoch data)
baseline_xs, baseline_ys = compute_one_epoch_baseline(all_datasets)

if parap_datasets:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6),
                                             sharex=True, sharey=True)
    plot_panel(ax_left, all_datasets, baseline_xs, baseline_ys,
               title='Multi-epoch', axis_field='epochs')
    plot_panel(ax_right, parap_datasets, baseline_xs, baseline_ys,
               title='Paraphrase', axis_field='K')
    # Shared x-limits derived from union of all flops on the figure
    all_flops = list(baseline_xs)
    for d in all_datasets: all_flops += list(d['flops_multiplier'])
    for d in parap_datasets: all_flops += list(d['flops_multiplier'])
    ax_left.set_xlim(min(all_flops) * 0.7, max(all_flops) * 1.3)
else:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_panel(ax, all_datasets, baseline_xs, baseline_ys,
               title=None, axis_field='epochs')
    all_flops = list(baseline_xs)
    for d in all_datasets: all_flops += list(d['flops_multiplier'])
    ax.set_xlim(min(all_flops) * 0.7, max(all_flops) * 1.3)

plt.tight_layout()
out_path = f'results/compute_optimal_scaling_{MODEL_SIZE}.pdf'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')
plt.show()
