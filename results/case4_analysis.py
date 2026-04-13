import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# ---- Select model size: '30m' or '370m' ----
MODEL_SIZE = '30m'

if MODEL_SIZE == '30m':
    import data_30m as model_data
    self_distill_data = model_data.self_distill_data
    parap_best_data = model_data.parap_best_data
elif MODEL_SIZE == '370m':
    import data_370m as model_data
    self_distill_data = None
    parap_best_data = None
else:
    raise ValueError(f'Unknown MODEL_SIZE: {MODEL_SIZE}')

all_datasets = model_data.ALL_DATASETS

# Build labeled dataset list for plotting
_ttp_labels = {
    0.05: 'TTP=1 (0.05x Chinchilla)',
    0.1:  'TTP=2 (0.1x Chinchilla)',
    0.5:  'TTP=10 (0.5x Chinchilla)',
    1:    'TTP=20 (1x Chinchilla)',
    2:    'TTP=40 (2x Chinchilla)',
    4:    'TTP=80 (4x Chinchilla)',
    8:    'TTP=160 (8x Chinchilla)',
    16:   'TTP=320 (16x Chinchilla)',
}
datasets = [
    (d, _ttp_labels.get(d['chinchilla_scale'][0], f"{d['chinchilla_scale'][0]}x Chinchilla"), 'o')
    for d in all_datasets
]

# Get unique chinchilla scales for colormap
chinchilla_values = sorted(d['chinchilla_scale'][0] for d in all_datasets)
norm = plt.Normalize(vmin=-1, vmax=16)
cmap = plt.cm.magma_r

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

for idx, (data, label, marker) in enumerate(datasets):
    flops = np.array(data['flops_multiplier'], dtype=float)
    loss = np.array(data['validation_loss'], dtype=float)
    chinchilla = data['chinchilla_scale'][0]

    # Filter out NaN values
    valid_mask = ~np.isnan(loss)
    flops_valid = flops[valid_mask]
    loss_valid = loss[valid_mask]

    color = cmap(norm(chinchilla))
    # Plot line connecting valid points, then plot markers
    ax.plot(flops_valid, loss_valid, linestyle='-', marker=marker, color=color,
            label=label, linewidth=2, markersize=5, zorder=10-idx)

# Add dashed line connecting 1-epoch points across all chinchilla setups
one_epoch_flops = []
one_epoch_loss = []
for data, label, marker in datasets:
    epochs = np.array(data['epochs'])
    flops = np.array(data['flops_multiplier'], dtype=float)
    loss = np.array(data['validation_loss'], dtype=float)
    # Find the index where epoch == 1
    idx_1ep = np.where(epochs == 1)[0]
    if len(idx_1ep) > 0 and not np.isnan(loss[idx_1ep[0]]):
        one_epoch_flops.append(flops[idx_1ep[0]])
        one_epoch_loss.append(loss[idx_1ep[0]])

ax.plot(one_epoch_flops, one_epoch_loss, linestyle='--', color='black',
        linewidth=4, alpha=0.5, label='Compute Optimal (1-epoch)', zorder=15)

ax.set_xlabel('FLOPs (Chinchilla Optimal = 1X)', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title(f'Validation Loss vs FLOPs for Different Chinchilla Scales ({MODEL_SIZE.upper()})', fontsize=14)
ax.set_xscale('log', base=2)
ax.set_xlim(0.03, 180)
# ax.set_ylim(3.8, 8.0)

# Format x-axis ticks as plain numbers (0.05, 0.5, 1, 2, 4, etc.)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:g}'))

ax.legend(fontsize=10, title='Data Size (Chinchilla Scale)', title_fontsize=11)
ax.grid(True, alpha=0.3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Chinchilla Scale', fontsize=12)

plt.tight_layout()
plt.savefig(f'results/case4_flops_vs_loss_{MODEL_SIZE}.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Second plot: Data Size vs Loss, scatter colored by FLOPs ----

# Collect all points as flat arrays
all_chin = []
all_loss = []
all_flops = []

for data in all_datasets:
    chin = data['chinchilla_scale'][0]
    for f, l in zip(data['flops_multiplier'], data['validation_loss']):
        if not np.isnan(l):
            all_chin.append(chin)
            all_loss.append(l)
            all_flops.append(f)

all_chin = np.array(all_chin)
all_loss = np.array(all_loss)
all_flops = np.array(all_flops)

# Convert chinchilla scale to TTP (token-to-param ratio) = chinchilla_scale * 20
all_ttp = all_chin * 20

# Colormap for FLOPs values (log scale)
norm2 = plt.Normalize(vmin=np.log2(all_flops.min()), vmax=np.log2(all_flops.max()))
cmap2 = plt.cm.viridis

fig2, ax2 = plt.subplots(figsize=(10, 6))

sc = ax2.scatter(all_ttp, all_loss, c=np.log2(all_flops), cmap=cmap2, norm=norm2,
                 s=50, edgecolors='k', linewidths=0.5, zorder=10)

ax2.set_xlabel('Data Size (TTP, Chinchilla X)', fontsize=12)
ax2.set_ylabel('Validation Loss', fontsize=12)
ax2.set_title(f'Validation Loss vs Data Size for Different FLOPs Budgets ({MODEL_SIZE.upper()})', fontsize=14)
ax2.set_xscale('log', base=2)
ax2.set_xlim(0.04 * 20, 18 * 20)

# TTP tick values with chinchilla scale in parentheses
chinchilla_values = sorted(set(all_chin))
ttp_values = [c * 20 for c in chinchilla_values]
ax2.set_xticks(ttp_values)
ax2.xaxis.set_major_formatter(FuncFormatter(
    lambda x, p: f'{x:g} ({x/20:g}x)'))
ax2.xaxis.set_minor_formatter(plt.NullFormatter())

ax2.grid(True, alpha=0.3)

# Envelope: connect the lowest loss at each data scale
chin_unique = sorted(set(all_chin))
min_loss_per_chin = [all_loss[all_chin == c].min() for c in chin_unique]
ax2.plot([c * 20 for c in chin_unique], min_loss_per_chin, linestyle='--', color='black',
         linewidth=2.5, alpha=0.7, label='Data Optimal (w/ multi-epoch)', zorder=15)

# Compute Optimal: connect 1-epoch points across all data scales
compute_opt_chin = []
compute_opt_loss = []
for data in all_datasets:
    epochs = np.array(data['epochs'])
    loss = np.array(data['validation_loss'], dtype=float)
    idx_1ep = np.where(epochs == 1)[0]
    if len(idx_1ep) > 0 and not np.isnan(loss[idx_1ep[0]]):
        compute_opt_chin.append(data['chinchilla_scale'][0])
        compute_opt_loss.append(loss[idx_1ep[0]])
ax2.plot([c * 20 for c in compute_opt_chin], compute_opt_loss, linestyle='--', color='tab:orange',
         linewidth=2.5, alpha=0.7, label='Compute Optimal (1-epoch)', zorder=15)

# Self-distillation and paraphrasing points (30M only)
if self_distill_data is not None:
    ax2.scatter([c * 20 for c in self_distill_data['chinchilla_scale']],
                self_distill_data['validation_loss'],
                marker='X', c='red', s=120, edgecolors='k', linewidths=0.5,
                label='Self-distillation', zorder=20)

if parap_best_data is not None:
    ax2.scatter([c * 20 for c in parap_best_data['chinchilla_scale']],
                parap_best_data['validation_loss'],
                marker='D', c='cornflowerblue', s=120, edgecolors='k', linewidths=0.5,
                label='Paraphrasing', zorder=19)

ax2.legend(fontsize=10)

# Add colorbar with FLOPs tick labels
cbar2 = plt.colorbar(sc, ax=ax2)
cbar2.set_label('FLOPs (Chinchilla Optimal = 1X)', fontsize=12)
flops_ticks = [0.05, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
cbar2.set_ticks([np.log2(v) for v in flops_ticks])
cbar2.set_ticklabels([f'{v:g}' for v in flops_ticks])

plt.tight_layout()
plt.savefig(f'results/case4_datasize_vs_loss_{MODEL_SIZE}.png', dpi=150, bbox_inches='tight')
plt.show()