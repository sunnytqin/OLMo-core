import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05],
    'epochs': [1],
    'flops_multiplier': [0.05],
    'validation_loss': [7.618],
    'learning_rate': [1e-3],
    'weight_decay': [0.1],
}

data_0_1x = {
    'chinchilla_scale': [0.1],
    'epochs': [1],
    'flops_multiplier': [0.1],
    'validation_loss': [6.8242],
    'learning_rate': [1e-3],
    'weight_decay': [0.1],
}

data_0_25x = {
    'chinchilla_scale': [0.25],
    'epochs': [1],
    'flops_multiplier': [0.25],
    'validation_loss': [5.8502],
    'learning_rate': [1e-3],
    'weight_decay': [0.1],
}

data_0_5x = {
    'chinchilla_scale': [0.5],
    'epochs': [1],
    'flops_multiplier': [0.5],
    'validation_loss': [4.8499],
    'learning_rate': [1e-3],
    'weight_decay': [0.1],
}

ALL_DATASETS = [
    data_0_05x,
    data_0_1x,
    data_0_25x,
    data_0_5x,
]
