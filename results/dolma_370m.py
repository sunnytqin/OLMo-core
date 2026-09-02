import numpy as np

data_0_05x = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    'validation_loss': [6.0294, 5.1426, 3.9614, 3.4115, 3.2618, 3.3905, 3.4884],
    'learning_rate': [1e-3, 1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.4, 0.2, 0.1, 0.8, 1.6, 1.6],
}

data_0_1x = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'epochs': [1, 2, 4, 8, 16, 32, 64],
    'flops_multiplier': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    'validation_loss': [5.1539, 4.0313, 3.3384, 3.0968, 3.0115, 3.0404, 3.0026],
    'learning_rate': [1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.4, 0.2, 0.8, 0.8, 0.4],
}

data_0_25x = {
    'chinchilla_scale': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    'validation_loss': [3.6723, 3.1607, 2.9488, 2.8186, 2.7617, 2.746],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.4, 0.4, 0.4],
}

data_0_5x = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'epochs': [1, 2, 4, 8, 16, 32],
    'flops_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'validation_loss': [3.1976, 2.9057, 2.7548, 2.6704, 2.6763, 2.6602],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.2, 0.4, 0.4],
}

data_1x = {
    'chinchilla_scale': [1, 1, 1, 1, 1],
    'epochs': [1, 2, 4, 8, 16],
    'flops_multiplier': [1, 2, 4, 8, 16],
    'validation_loss': [2.9141, 2.7418, 2.6505, 2.6048, 2.5835],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.2, 0.2],
}

data_2x = {
    'chinchilla_scale': [2, 2, 2, 2],
    'epochs': [1, 2, 4, 8],
    'flops_multiplier': [2, 4, 8, 16],
    'validation_loss': [2.7294, 2.6353, 2.5946, 2.5702],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.2, 0.2, 0.2],
}

data_4x = {
    'chinchilla_scale': [4],
    'epochs': [1],
    'flops_multiplier': [4],
    'validation_loss': [2.6374],
    'learning_rate': [3e-3],
    'weight_decay': [0.2],
}

data_8x = {
    'chinchilla_scale': [8],
    'epochs': [1],
    'flops_multiplier': [8],
    'validation_loss': [2.5877],
    'learning_rate': [3e-3],
    'weight_decay': [0.2],
}

data_0_05x_para = {
    'chinchilla_scale': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'K': [1, 2, 4, 8, 16, 32, 64],
    'tokens_trained': [483944386, 607413209, 851040456, 1334907447, 2322537946, 4166006910, 7926159870],
    'flops_multiplier': [0.0654, 0.0821, 0.115, 0.1804, 0.3139, 0.563, 1.0711],
    'validation_loss': [6.056, 5.768, 5.2947, 4.2886, 3.7609, 3.3389, 3.1464],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_1x_para = {
    'chinchilla_scale': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'K': [1, 2, 4, 8, 16, 32, 64],
    'tokens_trained': [968029964, 1215160023, 1703151354, 2670192619, 4650806361, 8342892259, 15872392809],
    'flops_multiplier': [0.1308, 0.1642, 0.2302, 0.3608, 0.6285, 1.1274, 2.1449],
    'validation_loss': [4.9194, 4.4847, 3.9688, 3.5881, 3.2884, 3.1032, 2.9634],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_0_5x_para = {
    'chinchilla_scale': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'K': [1, 2, 4, 8, 16, 32],
    'tokens_trained': [4840489570, 6076056272, 8525356437, 13378912780, 23317234568, 41820003052],
    'flops_multiplier': [0.6541, 0.8211, 1.1521, 1.808, 3.151, 5.6514],
    'validation_loss': [3.1342, 3.0855, 3.0187, 2.9176, 2.8309, 2.7719],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

data_1x_para = {
    'chinchilla_scale': [1, 1, 1, 1, 1, 1],
    'K': [1, 2, 4, 8, 16, 32],
    'tokens_trained': [9670419295, 12131079742, 17009424061, 26679032762, 46482745254, 83338966201],
    'flops_multiplier': [1.3068, 1.6393, 2.2986, 3.6053, 6.2815, 11.262],
    'validation_loss': [2.903, 2.8626, 2.838, 2.7853, 2.7372, 2.713],
    'learning_rate': [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3],
    'weight_decay': [0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
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
    data_0_5x_para,
    data_1x_para,
]

selfdistill_datasets = None
