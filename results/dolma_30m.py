import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [8.1971, 8.0091, 7.5634, 6.675, 5.9493, 5.4662, 5.3387, 5.7203],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.1, 0.2, 0.4, 1.6, 1.6],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
    'validation_loss': [8.0118, 7.5623, 6.6649, 5.9173, 5.2971, 4.8047, 4.6177, 4.8208],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.4, 0.1, 0.1, 0.2, 0.8, 0.8],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [7.2161, 6.5023, 5.6828, 4.9073, 4.3311, 4.1665, 4.1152],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.1, 0.2, 0.1, 0.4, 0.8, 0.8],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    'validation_loss': [6.4435, 5.6401, 4.8736, 4.2115, 3.9728, 3.8961, 3.882],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.4],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1],
    'epochs': [1, 4, 8, 16, 32, 64],
    'flops_multiplier': [1, 4, 8, 16, 32, 64],
    'validation_loss': [5.6831, 4.2052, 3.9541, 3.8017, 3.7418, 3.7049],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.4, 0.2, 0.2, 0.1, 0.1, 0.1],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2, 2, 2],
    'epochs': [1, 4, 8, 16, 32, 64],
    'flops_multiplier': [2, 8, 16, 32, 64, 128],
    'validation_loss': [4.9317, 3.8841, 3.7551, 3.7005, 3.6463, 3.6069],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
}

data_4x = {
    'chinchilla_scale': [4, 4, 4, 4, 4],
    'epochs': [1, 4, 8, 16, 32],
    'flops_multiplier': [4, 16, 32, 64, 128],
    'validation_loss': [4.1337, 3.7851, 3.659, 3.6304, 3.5732],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.4, 0.1, 0.2, 0.1],
}

data_8x = {
    'chinchilla_scale': [8, 8],
    'epochs': [1, 2],
    'flops_multiplier': [8, 16],
    'validation_loss': [3.8764, 3.7343],
    'learning_rate': [3e-3, 3e-3],
    'weight_decay': [0.1, 0.1],
}

data_16x = {
    'chinchilla_scale': [16],
    'epochs': [1],
    'flops_multiplier': [16],
    'validation_loss': [3.7454],
    'learning_rate': [3e-3],
    'weight_decay': [0.1],
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
    data_16x,
]
