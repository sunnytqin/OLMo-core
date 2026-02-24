import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- Case 3: Paraphrased + Multi-epoch ----

parap_data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [7.78, 7.71, 7.18, 6.12, 5.64, 5.10, 4.98, 4.98],
    'learning_rate': [0.01, 0.003, 0.003, 0.06, 0.03, 0.03, 0.006, 0.003],
    'weight_decay': [1.6, 0.1, 0.1, 1.6, 0.8, 0.8, 0.8, 1.6]
}

parap_data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
    'validation_loss': [7.69, 7.15, 6.09, 5.55, 4.96, 4.61, 4.56, 4.65],
    'learning_rate': [0.003, 0.003, 0.06, 0.06, 0.03, 0.06, 0.01, 0.01],
    'weight_decay': [0.1, 0.1, 1.6, 0.4, 0.4, 0.1, 0.4, 0.8]
}

parap_data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.5, 1, 2, 4, 8, 16, 32, 64],
    'validation_loss': [6.05, 5.34, 4.68, 4.30, 4.19, 4.15, 4.11, 4.11],
    'learning_rate': [0.06, 0.06, 0.06, 0.03, 0.03, 0.01, 0.01, 0.006],
    'weight_decay': [0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01]
}

parap_data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1, 1],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [1, 2, 4, 8, 16, 32, 64],
    'validation_loss': [5.57, 4.89, 4.29, 4.15, 4.10, 4.05, 4.04],
    'learning_rate': [0.06, 0.06, 0.03, 0.03, 0.03, 0.01, 0.003],
    'weight_decay': [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
}

# ---- Case 4: Plain Multi-epoch (no augmentation) ----

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
    'flops_multiplier': [0.5, 1, 2, 4, 8, 16, 32, 64],
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

# ---- Comparison plot: Case 3 (Paraphrased) vs Case 4 (Plain) ----

# Colors for each data size
colors = {0.05: 'tab:blue', 0.1: 'tab:orange', 0.5: 'tab:green', 1: 'tab:red'}

ttp_labels = {0.05: '1', 0.1: '2', 0.5: '10', 1: '20'}

pairs = [
    (data_0_05x, parap_data_0_05x, 0.05),
    (data_0_1x, parap_data_0_1x, 0.1),
    (data_0_5x, parap_data_0_5x, 0.5),
    (data_1x, parap_data_1x, 1),
]

fig, ax = plt.subplots(figsize=(10, 6))

for plain, parap, chin_scale in pairs:
    flops = np.array(plain['flops_multiplier'], dtype=float)
    loss_plain = np.array(plain['validation_loss'], dtype=float)
    loss_parap = np.array(parap['validation_loss'], dtype=float)
    color = colors[chin_scale]
    ttp = ttp_labels[chin_scale]

    # Plain multi-epoch (case 4) — solid line
    ax.plot(flops, loss_plain, linestyle='-', marker='o', color=color,
            label=f'Data Size = {ttp}TTP; Multi-epoch', linewidth=2, markersize=5)
    # Paraphrased + multi-epoch (case 3) — dashed line
    ax.plot(flops, loss_parap, linestyle='--', marker='s', color=color,
            label=f'Data Size = {ttp}TTP; Paraphrased+Multi-epoch', linewidth=2, markersize=5)

ax.set_xlabel('FLOPs (Chinchilla Optimal = 1X)', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Multi-epoch vs Paraphrased+Multi-epoch Training', fontsize=14)
ax.set_xscale('log', base=2)
ax.set_xlim(0.03, 180)
ax.set_ylim(3.8, 8.0)

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:g}'))
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/case3_flops_vs_loss.png', dpi=150, bbox_inches='tight')
plt.show()
