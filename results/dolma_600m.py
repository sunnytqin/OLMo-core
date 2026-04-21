import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05],
    'epochs': [1],
    'flops_multiplier': [0.05],
    'validation_loss': [5.4726],
    'learning_rate': [1e-3],
    'weight_decay': [0.1],
}

data_0_1x = {
    'chinchilla_scale': [0.1],
    'epochs': [1],
    'flops_multiplier': [0.1],
    'validation_loss': [4.1604],
    'learning_rate': [1e-3],
    'weight_decay': [0.1],
}

ALL_DATASETS = [
    data_0_05x,
    data_0_1x,
]
