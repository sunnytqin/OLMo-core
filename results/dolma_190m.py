import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [6.6618, 5.952, 5.146, 4.1333, 3.5928, 3.7211, 3.8629],
    'learning_rate': [1e-3, 1e-3, 1e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.4, 0.8, 1.6, 1.6],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [6.0518, 5.1828, 4.097, 3.4631, 3.2827, 3.4115],
    'learning_rate': [1e-3, 1e-3, 1e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.4, 0.8, 1.6],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    'validation_loss': [4.8364, 3.7622, 3.3117, 3.0864, 3.0206, 3.0368],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.4, 0.2, 0.2],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [3.8272, 3.2832, 3.0462, 2.9249, 2.8671, 2.8732],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2, 0.2, 0.1],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [1, 2, 4, 8, 16, 32],
    'validation_loss': [3.331, 3.039, 2.8827, 2.8502, 2.7779, 2.7574],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.4, 0.1, 0.1],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [2, 4, 8, 16],
    'validation_loss': [3.0447, 2.8741, 2.7999, 2.7695],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2],
}

data_4x = {
    'chinchilla_scale': [4, 4, 4],
    'epochs': [1, 2, 4],
    'flops_multiplier': [4, 8, 16],
    'validation_loss': [2.8778, 2.7896, 2.7381],
    'learning_rate': [3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1],
}

data_8x = {
    'chinchilla_scale': [8, 8],
    'epochs': [1, 2],
    'flops_multiplier': [8, 16],
    'validation_loss': [2.7867, 2.7408],
    'learning_rate': [3e-3, 3e-3],
    'weight_decay': [0.1, 0.1],
}

data_16x = {
    'chinchilla_scale': [16, 16],
    'epochs': [1, 2],
    'flops_multiplier': [16, 32],
    'validation_loss': [2.7356, 2.7047],
    'learning_rate': [3e-3, 3e-3],
    'weight_decay': [0.1, 0.1],
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
