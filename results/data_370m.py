import numpy as np

# D=0.05Chinchilla (1 TTP)
# Note: 64-epoch entry was not fully evaluated
data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [6.02, 5.13, 3.94, 3.39, 3.24, 3.37, np.nan],
    'learning_rate': [0.001, 0.001, 0.003, 0.003, 0.003, 0.003, 0.003],
    'weight_decay': [0.2, 0.4, 0.2, 0.1, 0.8, 1.6, 1.6],
}

# D=0.1Chinchilla (2 TTP)
# Note: 4-epoch crashed 2/3 of the way (needs rerun); 64-epoch pending
data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [5.14, 4.01, np.nan, 3.08, 2.96, 2.98, np.nan],
    'learning_rate': [0.001, 0.003, 0.001, 0.003, 0.003, 0.003, np.nan],
    'weight_decay': [0.1, 0.1, 0.8, 0.2, 0.8, 0.8, np.nan],
}

ALL_DATASETS = [
    data_0_05x,
    data_0_1x,
]
