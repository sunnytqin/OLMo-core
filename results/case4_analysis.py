import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

# Collect all datasets with markers
datasets = [
    (data_1x, '1x Chinchilla', 'o'),
    (data_2x, '2x Chinchilla', 'o'),
    (data_4x, '4x Chinchilla', 'o'),
    (data_8x, '8x Chinchilla', 'o'),
    (data_16x, '16x Chinchilla', 'o'),
]

# Get unique chinchilla scales for colormap
chinchilla_values = [1, 4, 8, 16]
norm = plt.Normalize(vmin=0, vmax=32)
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
        linewidth=4, alpha=0.5, label='1 Epoch', zorder=15)

ax.set_xlabel('FLOPs (Chinchilla Optimal = 1X)', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Validation Loss vs FLOPs for Different Chinchilla Scales', fontsize=14)
ax.set_xscale('log', base=2)
ax.set_yscale('log')

# Format x-axis ticks as plain numbers (1, 2, 4, etc.)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(False)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Chinchilla Scale', fontsize=12)

plt.tight_layout()
plt.savefig('results/case4_flops_vs_loss.png', dpi=150, bbox_inches='tight')
plt.show()