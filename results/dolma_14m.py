import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [10.2806, 9.2009, 8.2162, 7.8159, 7.0861, 6.4091, 6.1188, 6.3766],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 1.6],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64, 128],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
    'validation_loss': [9.2142, 8.2144, 7.8615, 7.1053, 6.387, 5.7526, 5.6692, 5.736],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.4, 0.1, 0.4, 0.8, 1.6],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [8.0745, 7.6403, 6.8801, 6.0321, 5.409, 5.0083, 4.8192],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [7.7081, 6.9261, 6.0471, 5.3494, 4.7967, 4.5544],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.1, 0.1, 0.4],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [1, 2, 4, 8, 16, 32],
    'validation_loss': [6.8567, 6.0532, 5.3521, 4.7893, 4.4627, 4.3076],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.1, 0.1, 0.2],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2, 2, 2],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [2, 4, 8, 16, 32, 64],
    'validation_loss': [6.0082, 5.3081, 4.7446, 4.4087, 4.2379, 4.1815],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
}

data_4x = {
    'chinchilla_scale': [4, 4, 4, 4, 4],
    'epochs': [1, 2, 4, 8, 16],
    'flops_multiplier': [4, 8, 16, 32, 64],
    'validation_loss': [5.3075, 4.7007, 4.4082, 4.2228, 4.1109],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1],
}

data_8x = {
    'chinchilla_scale': [8, 8, 8, 8, 8],
    'epochs': [1, 2, 4, 8, 16],
    'flops_multiplier': [8, 16, 32, 64, 128],
    'validation_loss': [4.7329, 4.3909, 4.2248, 4.0934, 4.0389],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1],
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
