import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [7.76, 7.71, 7.17, 6.06, 5.57, 5.19, 5.14, 5.18],
    'learning_rate': [0.02, 0.002, 0.003, 0.06, 0.06, 0.03, 0.01, 0.03],
    'weight_decay': [1.6, 0.1, 0.4, 1.6, 0.4, 0.8, 0.8, 1.6]
}   

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
    'validation_loss': [7.69, 6.96, 6.04, 5.54, 4.97, 4.80, 4.72, 4.84],
    'learning_rate': [0.003, 0.06, 0.06, 0.03, 0.03, 0.03, 0.006, 0.003],
    'weight_decay': [0.1, 1.6, 1.6, 0.8, 0.4, 0.4, 0.4, 0.8]
}


data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.5, 1, 2,4, 8, 16, 32, 64],
    'validation_loss': [6.05, 5.27, 4.69, 4.24, 4.16, 4.13, 4.13, 4.14],
    'learning_rate': [0.06, 0.06, 0.06, 0.03, 0.03, 0.01, 0.01, 0.03],
    'weight_decay': [0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
}


data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1, 1],
    'epochs': [1, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [1, 4, 8, 16, 32, 64, 128],
    'validation_loss': [5.3, 4.39, 4.12, 4.05, 4.02, 4.00, 3.99],
    'learning_rate': [0.03, 0.01, 0.01, 0.01, 0.01, 0.006, 0.006],
    'weight_decay': [0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2, 2, 2, 2],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [2, 4, 8, 16, 32, 64, 128],
    'validation_loss': [4.61, 4.22, 4.11, 4.05, 3.99, 3.97, 3.94],
    'learning_rate': [0.06, 0.06, 0.03, 0.01, 0.01, 0.003, 0.003],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
}

data_4x = {
    'chinchilla_scale': [4, 4, 4, 4, 4, 4],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [4, 8, 16, 32, 64, 128],
    'validation_loss': [4.20, 4.08, 4.02, 3.97, 3.93, 3.92],
    'learning_rate': [0.06, 0.03, 0.01, 0.006, 0.006, 0.006],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
}

data_8x = {
    'chinchilla_scale': [8, 8, 8, 8, 8],
    'epochs': [1, 2, 4, 8, 16],
    'flops_multiplier': [8, 16, 32, 64, 128],
    'validation_loss': [4.07, 4.00, 3.96, 3.93, np.nan],
    'learning_rate': [0.03, 0.01, 0.01, 0.006, np.nan],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, np.nan]
}

data_16x = {
    'chinchilla_scale': [16, 16, 16, 16],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [16, 32, 64, 128],
    'validation_loss': [4.00, 3.96, 3.93, 3.89],
    'learning_rate': [0.03, 0.01, 0.003, 0.003],
    'weight_decay': [0.1, 0.1, 0.1, 0.1]
}

self_distill_data = {
    'chinchilla_scale': [0.05, 0.1, 0.5, 1],
    'validation_loss': [4.90, 4.52, 4.08, 3.98],
}

# Best paraphrasing loss at each data scale (from case 3)
parap_best_data = {
    'chinchilla_scale': [0.05, 0.1, 0.5, 1],
    'validation_loss': [4.98, 4.56, 4.11, 4.04],
}

# Collect all datasets with markers
datasets = [
    (data_0_05x, 'TTP=1 (0.05x Chinchilla)', 'o'),
    (data_0_1x, 'TTP=2 (0.1x Chinchilla)', 'o'),
    (data_0_5x, 'TTP=10 (0.5x Chinchilla)', 'o'),
    (data_1x, 'TTP=20 (1x Chinchilla)', 'o'),
    (data_2x, 'TTP=40 (2x Chinchilla)', 'o'),
    (data_4x, 'TTP=80 (4x Chinchilla)', 'o'),
    (data_8x, 'TTP=160 (8x Chinchilla)', 'o'),
    (data_16x, 'TTP=320 (16x Chinchilla)', 'o'),
]

# Get unique chinchilla scales for colormap
chinchilla_values = [0.05, 0.1, 0.5, 1, 2, 4, 8, 16]
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
ax.set_title('Validation Loss vs FLOPs for Different Chinchilla Scales', fontsize=14)
ax.set_xscale('log', base=2)
ax.set_xlim(0.03, 180)
ax.set_ylim(3.8, 8.0)

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
plt.savefig('results/case4_flops_vs_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Second plot: Data Size vs Loss, scatter colored by FLOPs ----

# Collect all points as flat arrays
all_datasets = [data_0_05x, data_0_1x, data_0_5x, data_1x, data_2x, data_4x, data_8x, data_16x]
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
ax2.set_title('Validation Loss vs Data Size for Different FLOPs Budgets', fontsize=14)
ax2.set_xscale('log', base=2)
ax2.set_xlim(0.04 * 20, 18 * 20)

# TTP tick values with chinchilla scale in parentheses
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

# Self-distillation points
ax2.scatter([c * 20 for c in self_distill_data['chinchilla_scale']],
            self_distill_data['validation_loss'],
            marker='X', c='red', s=120, edgecolors='k', linewidths=0.5,
            label='Self-distillation', zorder=20)

# Paraphrasing points (best loss at each data scale)
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
plt.savefig('results/case4_datasize_vs_loss.png', dpi=150, bbox_inches='tight')
plt.show()