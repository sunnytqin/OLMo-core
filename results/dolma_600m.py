import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4],
    'validation_loss': [5.4726, 4.5092, 3.3982, 3.1465],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.2],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8],
    'validation_loss': [4.1604, 3.3587, 3.0518, 2.8755],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.2],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0],
    'validation_loss': [3.2421, 2.9483, 2.7244, 2.6382],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.2],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0],
    'validation_loss': [2.9227, 2.7243, 2.6017, 2.5484],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.2],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1],
    'epochs': [1, 2, 4],
    'flops_multiplier': [1, 2, 4],
    'validation_loss': [2.7273, 2.5825, 2.5275],
    'learning_rate': [3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2],
}

data_2x = {
    'chinchilla_scale': [2, 2],
    'epochs': [1, 2],
    'flops_multiplier': [2, 4],
    'validation_loss': [2.5787, 2.5041],
    'learning_rate': [3e-3, 3e-3],
    'weight_decay': [0.1, 0.1],
}

data_4x = {
    'chinchilla_scale': [4],
    'epochs': [1],
    'flops_multiplier': [4],
    'validation_loss': [2.5021],
    'learning_rate': [3e-3],
    'weight_decay': [0.1],
}

data_0_05x_para = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'K': [1, 2, 4, 8, 16, 32],
    'tokens_trained': [785109027, 985803162, 1381771817, 2166836268, 3774431780, 6771984678],
    'flops_multiplier': [0.0654, 0.0822, 0.1151, 0.1806, 0.3145, 0.5643],
    'validation_loss': [5.3203, 4.9673, 4.1809, 3.7473, 3.3343, 3.0982],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_1x_para = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'K': [1, 2, 4, 8, 16, 32],
    'tokens_trained': [1570843038, 1971482440, 2765293587, 4336461741, 7554721060, 13550084872],
    'flops_multiplier': [0.1309, 0.1643, 0.2304, 0.3614, 0.6296, 1.1292],
    'validation_loss': [3.9501, 3.7331, 3.4362, 3.2443, 3.0587, 2.9189],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_25x_para = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25],
    'K': [1, 2, 4, 8, 16],
    'tokens_trained': [3923817347, 4924478833, 6907653182, 10835724186, 18882455812],
    'flops_multiplier': [0.327, 0.4104, 0.5756, 0.903, 1.5735],
    'validation_loss': [3.1769, 3.0928, 2.9832, 2.9363, 2.8058],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_5x_para = {
    'chinchilla_scale': [0.5, 0.5, 0.5],
    'K': [1, 2, 4],
    'tokens_trained': [7841057928, 9835694245, 13789147809],
    'flops_multiplier': [0.6534, 0.8196, 1.1491],
    'validation_loss': [2.8901, 2.8812, 2.8276],
    'learning_rate': [3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1],
}

ALL_DATASETS = [
    data_0_05x,
    data_0_1x,
    data_0_25x,
    data_0_5x,
    data_1x,
    data_2x,
    data_4x,
]

parap_datasets = [
    data_0_05x_para,
    data_0_1x_para,
    data_0_25x_para,
    data_0_5x_para,
]

selfdistill_datasets = None
