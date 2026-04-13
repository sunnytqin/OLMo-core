import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [7.76, 7.71, 7.17, 6.06, 5.57, 5.19, 5.14, 5.18],
    'learning_rate': [0.02, 0.002, 0.003, 0.06, 0.06, 0.03, 0.01, 0.03],
    'weight_decay': [1.6, 0.1, 0.4, 1.6, 0.4, 0.8, 0.8, 1.6],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
    'validation_loss': [7.69, 6.96, 6.04, 5.54, 4.97, 4.80, 4.72, 4.84],
    'learning_rate': [0.003, 0.06, 0.06, 0.03, 0.03, 0.03, 0.006, 0.003],
    'weight_decay': [0.1, 1.6, 1.6, 0.8, 0.4, 0.4, 0.4, 0.8],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.5, 1, 2, 4, 8, 16, 32, 64],
    'validation_loss': [6.05, 5.27, 4.69, 4.24, 4.16, 4.13, 4.13, 4.14],
    'learning_rate': [0.06, 0.06, 0.06, 0.03, 0.03, 0.01, 0.01, 0.03],
    'weight_decay': [0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1, 1],
    'epochs': [1, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [1, 4, 8, 16, 32, 64, 128],
    'validation_loss': [5.3, 4.39, 4.12, 4.05, 4.02, 4.00, 3.99],
    'learning_rate': [0.03, 0.01, 0.01, 0.01, 0.01, 0.006, 0.006],
    'weight_decay': [0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2, 2, 2, 2],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [2, 4, 8, 16, 32, 64, 128],
    'validation_loss': [4.61, 4.22, 4.11, 4.05, 3.99, 3.97, 3.94],
    'learning_rate': [0.06, 0.06, 0.03, 0.01, 0.01, 0.003, 0.003],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_4x = {
    'chinchilla_scale': [4, 4, 4, 4, 4, 4],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [4, 8, 16, 32, 64, 128],
    'validation_loss': [4.20, 4.08, 4.02, 3.97, 3.93, 3.92],
    'learning_rate': [0.06, 0.03, 0.01, 0.006, 0.006, 0.006],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_8x = {
    'chinchilla_scale': [8, 8, 8, 8, 8],
    'epochs': [1, 2, 4, 8, 16],
    'flops_multiplier': [8, 16, 32, 64, 128],
    'validation_loss': [4.07, 4.00, 3.96, 3.93, np.nan],
    'learning_rate': [0.03, 0.01, 0.01, 0.006, np.nan],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, np.nan],
}

data_16x = {
    'chinchilla_scale': [16, 16, 16, 16],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [16, 32, 64, 128],
    'validation_loss': [4.00, 3.96, 3.93, 3.89],
    'learning_rate': [0.03, 0.01, 0.003, 0.003],
    'weight_decay': [0.1, 0.1, 0.1, 0.1],
}

self_distill_data = {
    'chinchilla_scale': [0.05, 0.1, 0.5, 1],
    'validation_loss': [4.90, 4.52, 4.08, 3.98],
}

parap_best_data = {
    'chinchilla_scale': [0.05, 0.1, 0.5, 1],
    'validation_loss': [4.98, 4.56, 4.11, 4.04],
}

ALL_DATASETS = [
    data_0_05x,
    data_0_1x,
    data_0_5x,
    data_1x,
    data_2x,
    data_4x,
    data_8x,
    data_16x,
]
