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

data_0_25x = {
    'chinchilla_scale': [0.25],
    'epochs': [1],
    'flops_multiplier': [0.25],
    'validation_loss': [3.2421],
    'learning_rate': [3e-3],
    'weight_decay': [0.1],
}

data_0_5x = {
    'chinchilla_scale': [0.5],
    'epochs': [1],
    'flops_multiplier': [0.5],
    'validation_loss': [2.9227],
    'learning_rate': [3e-3],
    'weight_decay': [0.1],
}

data_1x = {
    'chinchilla_scale': [1],
    'epochs': [1],
    'flops_multiplier': [1],
    'validation_loss': [2.7273],
    'learning_rate': [3e-3],
    'weight_decay': [0.1],
}

data_2x = {
    'chinchilla_scale': [2],
    'epochs': [1],
    'flops_multiplier': [2],
    'validation_loss': [2.5787],
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
]
