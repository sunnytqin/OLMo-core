import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [6.0294, 5.1426, 3.9614, 3.4115, 3.2618, 3.3905, 3.4884],
    'learning_rate': [1e-3, 1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.4, 0.2, 0.1, 0.8, 1.6, 1.6],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [5.1539, 4.0313, 3.3384, 3.0968, 3.0115, 3.0404, 3.0026],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.4, 0.2, 0.8, 0.8, 0.4],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    'validation_loss': [3.6723, 3.1607, 2.9488, 2.8186, 2.7617, 2.746],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.4, 0.4, 0.4],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [3.1976, 2.9057, 2.7548, 2.6704, 2.6763, 2.6602],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.2, 0.4, 0.4],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1],
    'epochs': [1, 2, 4, 8, 16],
    'flops_multiplier': [1, 2, 4, 8, 16],
    'validation_loss': [2.9141, 2.7418, 2.6505, 2.6048, 2.5835],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.2, 0.2],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [2, 4, 8, 16],
    'validation_loss': [2.7294, 2.6353, 2.5946, 2.5702],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.2],
}

data_4x = {
    'chinchilla_scale': [4],
    'epochs': [1],
    'flops_multiplier': [4],
    'validation_loss': [2.6374],
    'learning_rate': [3e-3],
    'weight_decay': [0.2],
}

data_8x = {
    'chinchilla_scale': [8],
    'epochs': [1],
    'flops_multiplier': [8],
    'validation_loss': [2.5877],
    'learning_rate': [3e-3],
    'weight_decay': [0.2],
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
