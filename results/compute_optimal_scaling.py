import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# ---- Select model size: '30m', '370m', 'dolma_30m', 'dolma_60m', 'dolma_370m' ----
MODEL_SIZE = 'dolma_30m'

if MODEL_SIZE == '30m':
    import dclm_30m as model_data
elif MODEL_SIZE == '370m':
    import data_370m as model_data
elif MODEL_SIZE == 'dolma_30m':
    import dolma_30m as model_data
elif MODEL_SIZE == 'dolma_60m':
    import dolma_60m as model_data
elif MODEL_SIZE == 'dolma_370m':
    import dolma_370m as model_data
else:
    raise ValueError(f'Unknown MODEL_SIZE: {MODEL_SIZE}')

all_datasets = model_data.ALL_DATASETS

# Font sizes to match data_optimal_scaling.py
FONT_LABEL  = 18
FONT_TITLE  = 20
FONT_LEGEND = 15
FONT_TICK   = 16

# Build labeled dataset list for plotting
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
datasets = [
    (d, _ttp_labels.get(d['chinchilla_scale'][0], f"{d['chinchilla_scale'][0]}x Chin"), 'o')
    for d in all_datasets
]

# Colormap for the colored lines (no colorbar)
chinchilla_values = sorted(d['chinchilla_scale'][0] for d in all_datasets)
norm = plt.Normalize(vmin=-1, vmax=16)
cmap = plt.cm.magma_r

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# --- Compute Optimal (Chinchilla scaling) dashed line — plotted first so it's behind ---
one_epoch_flops = []
one_epoch_loss = []
for data, label, marker in datasets:
    epochs = np.array(data['epochs'])
    flops = np.array(data['flops_multiplier'], dtype=float)
    loss = np.array(data['validation_loss'], dtype=float)
    idx_1ep = np.where(epochs == 1)[0]
    if len(idx_1ep) > 0 and not np.isnan(loss[idx_1ep[0]]):
        one_epoch_flops.append(flops[idx_1ep[0]])
        one_epoch_loss.append(loss[idx_1ep[0]])

compute_line, = ax.plot(one_epoch_flops, one_epoch_loss,
                        linestyle='--', color='black',
                        linewidth=7, alpha=0.7,
                        label='Compute Optimal (Chinchilla scaling)',
                        zorder=2)

# --- Star at Chinchilla Optimal point (data_scale=1, epoch=1, FLOPs=1) ---
chin_star_loss = None
for data, label, marker in datasets:
    if data['chinchilla_scale'][0] == 1:
        epochs = np.array(data['epochs'])
        flops = np.array(data['flops_multiplier'], dtype=float)
        loss = np.array(data['validation_loss'], dtype=float)
        idx_1ep = np.where(epochs == 1)[0]
        if len(idx_1ep) > 0 and not np.isnan(loss[idx_1ep[0]]):
            chin_star_loss = loss[idx_1ep[0]]
        break
if chin_star_loss is not None:
    ax.plot(1.0, chin_star_loss, marker='*', color='black',
            markersize=20, zorder=20, linestyle='none')
    ax.annotate('Chinchilla Optimal',
                xy=(1.0, chin_star_loss),
                xytext=(1.5, chin_star_loss + 0.05),
                fontsize=FONT_LEGEND,
                color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# --- Colored lines per fresh data size ---
colored_handles = []
colored_labels = []
for idx, (data, label, marker) in enumerate(datasets):
    flops = np.array(data['flops_multiplier'], dtype=float)
    loss = np.array(data['validation_loss'], dtype=float)
    chinchilla = data['chinchilla_scale'][0]

    valid_mask = ~np.isnan(loss)
    flops_valid = flops[valid_mask]
    loss_valid = loss[valid_mask]

    color = cmap(norm(chinchilla))
    line, = ax.plot(flops_valid, loss_valid, linestyle='-', marker=marker, color=color,
                    label=label, linewidth=2, markersize=5, zorder=10 + idx)
    colored_handles.append(line)
    colored_labels.append(label)

ax.set_xlabel('FLOPs (Chinchilla Optimal = 1X)', fontsize=FONT_LABEL)
ax.set_ylabel('Validation Loss', fontsize=FONT_LABEL)
ax.tick_params(axis='both', labelsize=FONT_TICK)
ax.set_xscale('log', base=2)
ax.set_xlim(0.03, 180)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:g}'))
ax.grid(True, alpha=0.3)

# --- Legend 1: colored lines with title "Fresh Data Size D" ---
legend1 = ax.legend(colored_handles, colored_labels,
                    fontsize=FONT_LEGEND,
                    title='Fresh Data Size $D$',
                    title_fontsize=FONT_LEGEND,
                    loc='upper right')
ax.add_artist(legend1)

# --- Legend 2: Compute Optimal line in its own separate box ---
ax.legend(handles=[compute_line], labels=['Compute Optimal (Chinchilla)'],
          fontsize=FONT_LEGEND,
          loc='lower left')

plt.tight_layout()
out_path = f'results/compute_optimal_scaling_{MODEL_SIZE}.pdf'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')
plt.show()
