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

data_0_05x_para = {
    'chinchilla_scale': [0.05],
    'K': [1],
    'tokens_trained': [18050720],
    'flops_multiplier': [0.0645],
    'validation_loss': [10.9012],
    'learning_rate': [1e-3],
    'weight_decay': [0.1],
}

data_0_5x_para = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5],
    'K': [1, 2, 4, 8],
    'tokens_trained': [183029559, 229900222, 321959292, 504642377],
    'flops_multiplier': [0.6537, 0.8211, 1.1499, 1.8023],
    'validation_loss': [7.3265, 7.0777, 6.763, 6.2351],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1],
}

data_1x_para = {
    'chinchilla_scale': [1, 1, 1, 1],
    'K': [1, 2, 4, 8],
    'tokens_trained': [365571284, 458334183, 640913409, 1003895032],
    'flops_multiplier': [1.3056, 1.6369, 2.289, 3.5853],
    'validation_loss': [6.634, 6.38, 5.9036, 5.4743],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1],
}

data_2x_para = {
    'chinchilla_scale': [2, 2, 2, 2],
    'K': [1, 2, 4, 8],
    'tokens_trained': [732746714, 919813663, 1288937673, 2021648712],
    'flops_multiplier': [2.617, 3.285, 4.6033, 7.2202],
    'validation_loss': [5.7062, 5.5413, 5.2168, 4.8419],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1],
}

data_4x_para = {
    'chinchilla_scale': [4, 4, 4, 4],
    'K': [1, 2, 4, 8],
    'tokens_trained': [1465864333, 1839955805, 2580297981, 4046007454],
    'flops_multiplier': [5.2352, 6.5713, 9.2153, 14.45],
    'validation_loss': [5.1291, 4.8761, 4.6737, 4.5631],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2],
}

data_8x_para = {
    'chinchilla_scale': [8, 8, 8, 8],
    'K': [1, 2, 4, 8],
    'tokens_trained': [2928799929, 3674752185, 5152763016, 8080626679],
    'flops_multiplier': [10.46, 13.1241, 18.4027, 28.8594],
    'validation_loss': [4.6385, 4.5063, 4.4274, 4.2946],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1],
}

data_16x_para = {
    'chinchilla_scale': [16, 16, 16, 16],
    'K': [1, 2, 4, 8],
    'tokens_trained': [5855327886, 7344153900, 10297981152, 16151651180],
    'flops_multiplier': [20.9119, 26.2291, 36.7785, 57.6845],
    'validation_loss': [4.3479, 4.3072, 4.2331, 4.1785],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1],
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

parap_datasets = [
    data_0_05x_para,
    data_0_5x_para,
    data_1x_para,
    data_2x_para,
    data_4x_para,
    data_8x_para,
    data_16x_para,
]
