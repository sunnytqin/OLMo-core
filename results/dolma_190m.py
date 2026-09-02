import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [6.6618, 5.952, 5.146, 4.1333, 3.5928, 3.7211, 3.8629],
    'learning_rate': [1e-3, 1e-3, 1e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.4, 0.8, 1.6, 1.6],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [6.0518, 5.175, 4.0728, 3.4631, 3.2827, 3.4115],
    'learning_rate': [1e-3, 1e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.2, 0.4, 0.8, 1.6],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    'validation_loss': [4.8364, 3.7622, 3.2972, 3.0864, 3.0206, 3.0368],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.4, 0.4, 0.2, 0.2],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [3.7874, 3.2595, 3.0408, 2.9249, 2.8671, 2.8732],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [1, 2, 4, 8, 16, 32],
    'validation_loss': [3.331, 3.039, 2.8827, 2.8502, 2.7779, 2.7574],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.2, 0.4, 0.1, 0.1],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [2, 4, 8, 16],
    'validation_loss': [3.0447, 2.8741, 2.7999, 2.7695],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.2],
}

data_4x = {
    'chinchilla_scale': [4, 4, 4],
    'epochs': [1, 2, 4],
    'flops_multiplier': [4, 8, 16],
    'validation_loss': [2.8778, 2.7896, 2.7381],
    'learning_rate': [3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1],
}

data_8x = {
    'chinchilla_scale': [8, 8],
    'epochs': [1, 2],
    'flops_multiplier': [8, 16],
    'validation_loss': [2.7867, 2.7408],
    'learning_rate': [3e-3, 3e-3],
    'weight_decay': [0.1, 0.1],
}

data_16x = {
    'chinchilla_scale': [16, 16],
    'epochs': [1, 2],
    'flops_multiplier': [16, 32],
    'validation_loss': [2.7356, 2.7047],
    'learning_rate': [3e-3, 3e-3],
    'weight_decay': [0.1, 0.1],
}

data_0_05x_para = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'K': [1, 2, 4, 8, 16, 32, 64],
    'tokens_trained': [248027018, 310928395, 434890093, 680812848, 1183994745, 2121396393, 4033440585],
    'flops_multiplier': [0.0653, 0.0818, 0.1144, 0.1792, 0.3116, 0.5583, 1.0614],
    'validation_loss': [6.7252, 6.5174, 6.1769, 5.7162, 4.5648, 3.8442, 3.4991],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_1x_para = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'K': [1, 2, 4, 8, 16, 32, 64],
    'tokens_trained': [497070129, 623906381, 874142142, 1371258740, 2385793174, 4279911086, 8143033021],
    'flops_multiplier': [0.1308, 0.1642, 0.23, 0.3609, 0.6278, 1.1263, 2.1429],
    'validation_loss': [6.0126, 5.8853, 5.3579, 4.4639, 3.871, 3.4217, 3.2317],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_25x_para = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'K': [1, 2, 4, 8, 16, 32, 64],
    'tokens_trained': [1242496804, 1558937564, 2184754952, 3424121617, 5964513017, 10695630428, 20348914658],
    'flops_multiplier': [0.327, 0.4102, 0.5749, 0.9011, 1.5696, 2.8146, 5.355],
    'validation_loss': [4.3141, 4.0933, 3.7386, 3.5314, 3.2838, 3.1229, 3.0399],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_5x_para = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'K': [1, 2, 4, 8, 16, 32, 64],
    'tokens_trained': [2486222768, 3120317325, 4376935984, 6866320009, 11964512570, 21456735649, 40831322080],
    'flops_multiplier': [0.6543, 0.8211, 1.1518, 1.8069, 3.1486, 5.6465, 10.7451],
    'validation_loss': [3.5815, 3.4831, 3.3724, 3.1983, 3.0734, 2.997, 2.9377],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
}

data_1x_para = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1],
    'K': [1, 2, 4, 8, 16, 32],
    'tokens_trained': [4970147247, 6237612767, 8750277711, 13729915768, 23926366715, 42908146143],
    'flops_multiplier': [1.3079, 1.6415, 2.3027, 3.6131, 6.2964, 11.2916],
    'validation_loss': [3.232, 3.1645, 3.0883, 3.0305, 2.9575, 2.9107],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.2, 0.2, 0.2, 0.1, 0.1],
}

data_2x_para = {
    'chinchilla_scale': [2, 2, 2, 2, 2, 2],
    'K': [1, 2, 4, 8, 16, 32],
    'tokens_trained': [9670419295, 12131079742, 17009424061, 26679032762, 46482745254, 83338966201],
    'flops_multiplier': [2.5448, 3.1924, 4.4762, 7.0208, 12.2323, 21.9313],
    'validation_loss': [3.014, 2.9808, 2.9518, 2.9227, 2.879, 2.8585],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
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
]

selfdistill_datasets = None
