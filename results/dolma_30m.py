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
    'chinchilla_scale': [8, 8, 8, 8, 8],
    'epochs': [1, 2, 4, 8, 16],
    'flops_multiplier': [8, 16, 32, 64, 128],
    'validation_loss': [3.8764, 3.7343, 3.6493, 3.5995, 3.5642],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1],
}

data_16x = {
    'chinchilla_scale': [16, 16, 16, 16, 16],
    'epochs': [1, 2, 4, 8, 16],
    'flops_multiplier': [16, 32, 64, 128, 256],
    'validation_loss': [3.7454, 3.6549, 3.5926, 3.5629, 3.5457],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_05x_para = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05],
    'K': [1, 2, 4, 8],
    'tokens_trained': [39175842, 49337008, 69148369, 108521507],
    'flops_multiplier': [0.0653, 0.0822, 0.1152, 0.1809],
    'validation_loss': [8.1116, 8.0923, 7.927, 7.6174],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.1, 0.2],
}

data_0_1x_para = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1],
    'K': [1, 2, 4, 8],
    'tokens_trained': [78469544, 98757247, 138509525, 217276871],
    'flops_multiplier': [0.1308, 0.1646, 0.2308, 0.3621],
    'validation_loss': [7.9215, 7.6659, 7.3115, 6.8166],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.1],
}

data_0_25x_para = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25],
    'K': [1, 2, 4, 8],
    'tokens_trained': [196180701, 246460684, 345235508, 541210823],
    'flops_multiplier': [0.327, 0.4108, 0.5754, 0.902],
    'validation_loss': [6.9562, 6.6982, 6.2638, 5.7962],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.1, 0.2],
}

data_0_5x_para = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5],
    'K': [1, 2, 4, 8],
    'tokens_trained': [392229264, 492300911, 689387446, 1081094707],
    'flops_multiplier': [0.6537, 0.8205, 1.149, 1.8018],
    'validation_loss': [6.1489, 5.85, 5.575, 5.0603],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.1],
}

data_1x_para = {
    'chinchilla_scale': [1, 1, 1, 1],
    'K': [1, 2, 4, 8],
    'tokens_trained': [785109027, 985803162, 1381771817, 2166836268],
    'flops_multiplier': [1.3085, 1.643, 2.303, 3.6114],
    'validation_loss': [5.388, 5.1273, 4.6489, 4.275],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.2, 0.1],
}

data_2x_para = {
    'chinchilla_scale': [2, 2, 2, 2],
    'K': [1, 2, 4, 8],
    'tokens_trained': [1570843038, 1971482440, 2765293587, 4336461741],
    'flops_multiplier': [2.6181, 3.2858, 4.6088, 7.2274],
    'validation_loss': [4.5164, 4.3329, 4.1125, 3.9939],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2],
}

data_4x_para = {
    'chinchilla_scale': [4, 4, 4, 4],
    'K': [1, 2, 4, 8],
    'tokens_trained': [3139428300, 3939920760, 5525520833, 8667602961],
    'flops_multiplier': [5.2324, 6.5665, 9.2092, 14.446],
    'validation_loss': [4.0514, 3.9715, 3.8848, 3.8209],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.1],
}

data_8x_para = {
    'chinchilla_scale': [8, 8, 8, 8],
    'K': [1, 2, 4, 8],
    'tokens_trained': [6273079931, 7868567196, 11032447728, 17302500551],
    'flops_multiplier': [10.4551, 13.1143, 18.3874, 28.8375],
    'validation_loss': [3.8305, 3.8112, 3.7595, 3.723],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.1, 0.1],
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

parap_datasets = [
    data_0_05x_para,
    data_0_1x_para,
    data_0_25x_para,
    data_0_5x_para,
    data_1x_para,
    data_2x_para,
    data_4x_para,
    data_8x_para,
]
