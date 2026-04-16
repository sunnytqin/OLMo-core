import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [8.0672, 7.4312, 6.72, 5.977, 5.211, 4.6546, 4.5379],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.1, 0.2, 0.4, 0.8],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [7.4437, 6.7208, 5.95, 5.2172, 4.3406, 4.0327, 4.138],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [6.4191, 5.7035, 4.8439, 4.0984, 3.7427, 3.6685, 3.761],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.4],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [5.7064, 4.9164, 4.0728, 3.6779, 3.5011, 3.4115],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.2, 0.4],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1],
    'epochs': [1, 4, 8, 16, 32, 64],
    'flops_multiplier': [1, 4, 8, 16, 32, 64],
    'validation_loss': [4.9059, 3.6322, 3.4454, 3.3374, 3.3004, 3.2723],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2, 0.4, 0.4],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2, 2],
    'epochs': [1, 4, 8, 16, 32],
    'flops_multiplier': [2, 8, 16, 32, 64],
    'validation_loss': [4.0581, 3.4269, 3.3075, 3.2443, 3.233],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2, 0.4],
}

data_4x = {
    'chinchilla_scale': [4, 4, 4, 4],
    'epochs': [1, 4, 8, 16],
    'flops_multiplier': [4, 16, 32, 64],
    'validation_loss': [3.6284, 3.301, 3.225, 3.1852],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2],
}

data_8x = {
    'chinchilla_scale': [8, 8, 8],
    'epochs': [1, 4, 8],
    'flops_multiplier': [8, 32, 64],
    'validation_loss': [3.4235, 3.2159, 3.165],
    'learning_rate': [1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1],
}

ALL_DATASETS = [
    data_0_05x,
    data_0_1x,
    data_0_25x,
    data_0_5x,
    data_1x,
    data_2x,
    data_4x,
    data_8x,
]
