import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [8.0672, 7.4312, 6.72, 5.977, 5.211, 4.6546, 4.5379],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.1, 0.2, 0.4, 0.8],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [7.4437, 6.7208, 5.95, 5.2172, 4.3406, 4.0327, 4.138],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [6.4191, 5.7035, 4.8439, 4.0984, 3.7427, 3.6685, 3.6014],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    'validation_loss': [5.7064, 4.9164, 4.0728, 3.6779, 3.5011, 3.4115, 3.3789],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.4],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1],
    'epochs': [1, 4, 8, 16, 32, 64],
    'flops_multiplier': [1, 4, 8, 16, 32, 64],
    'validation_loss': [4.9059, 3.6322, 3.4454, 3.3374, 3.3004, 3.2723],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2, 0.4, 0.4],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2, 2, 2],
    'epochs': [1, 4, 8, 16, 32, 64],
    'flops_multiplier': [2, 8, 16, 32, 64, 128],
    'validation_loss': [4.0581, 3.4269, 3.3075, 3.2443, 3.233, 3.2116],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2, 0.4, 0.4],
}

data_4x = {
    'chinchilla_scale': [4, 4, 4, 4, 4],
    'epochs': [1, 4, 8, 16, 32],
    'flops_multiplier': [4, 16, 32, 64, 128],
    'validation_loss': [3.6284, 3.301, 3.225, 3.1852, 3.1916],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2, 0.4],
}

data_8x = {
    'chinchilla_scale': [8, 8, 8, 8],
    'epochs': [1, 4, 8, 16],
    'flops_multiplier': [8, 32, 64, 128],
    'validation_loss': [3.4235, 3.2159, 3.165, 3.147],
    'learning_rate': [1e-3, 1e-3, 1e-3, 1e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2],
}

data_0_05x_para = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05],
    'K': [1, 2, 4, 8, 16],
    'tokens_trained': [78469544, 98757247, 138509525, 217276871, 378153239],
    'flops_multiplier': [0.0654, 0.0823, 0.1154, 0.1811, 0.3151],
    'validation_loss': [8.0583, 7.9049, 7.4638, 6.9144, 6.6093],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.1, 0.1, 0.1, 0.1],
}

data_0_1x_para = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1],
    'K': [1, 2, 4, 8, 16],
    'tokens_trained': [157389299, 197962538, 277793755, 435874021, 759808811],
    'flops_multiplier': [0.1312, 0.165, 0.2315, 0.3632, 0.6332],
    'validation_loss': [7.3793, 7.1831, 6.7303, 6.2944, 5.5772],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.2, 0.1],
}

data_0_25x_para = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25],
    'K': [1, 2, 4, 8, 16],
    'tokens_trained': [392229264, 492300911, 689387446, 1081094707, 1881094955],
    'flops_multiplier': [0.3269, 0.4103, 0.5745, 0.9009, 1.5676],
    'validation_loss': [6.3038, 6.15, 5.657, 5.0172, 4.2417],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.2, 0.2, 0.1],
}

data_0_5x_para = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5],
    'K': [1, 2, 4, 8, 16],
    'tokens_trained': [785109027, 985803162, 1381771817, 2166836268, 3774431780],
    'flops_multiplier': [0.6543, 0.8215, 1.1515, 1.8057, 3.1454],
    'validation_loss': [5.3936, 5.115, 4.6183, 4.0777, 3.7359],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.1, 0.1, 0.1],
}

data_1x_para = {
    'chinchilla_scale': [1, 1, 1, 1, 1],
    'K': [1, 2, 4, 8, 16],
    'tokens_trained': [1570843038, 1971482440, 2765293587, 4336461741, 7554721060],
    'flops_multiplier': [1.309, 1.6429, 2.3044, 3.6137, 6.2956],
    'validation_loss': [4.4051, 4.0228, 3.8613, 3.6546, 3.4915],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.1, 0.1, 0.1],
}

data_2x_para = {
    'chinchilla_scale': [2, 2, 2, 2, 2],
    'K': [1, 2, 4, 8, 16],
    'tokens_trained': [3139428300, 3939920760, 5525520833, 8667602961, 15103210246],
    'flops_multiplier': [2.6162, 3.2833, 4.6046, 7.223, 12.586],
    'validation_loss': [3.8067, 3.6519, 3.5197, 3.437, 3.3599],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.1, 0.1, 0.1],
}

data_4x_para = {
    'chinchilla_scale': [4, 4, 4, 4, 4],
    'K': [1, 2, 4, 8, 16],
    'tokens_trained': [6273079931, 7868567196, 11032447728, 17302500551, 30144742585],
    'flops_multiplier': [5.2276, 6.5571, 9.1937, 14.4188, 25.1206],
    'validation_loss': [3.4452, 3.4071, 3.3637, 3.3159, 3.2906],
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

parap_datasets = [
    data_0_05x_para,
    data_0_1x_para,
    data_0_25x_para,
    data_0_5x_para,
    data_1x_para,
    data_2x_para,
    data_4x_para,
]

selfdistill_datasets = None
