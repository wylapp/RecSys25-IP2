import torch
import numpy as np
import os
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def update_config(config):
    lang = config.lang
    keys = ['json_path']
    for k in keys:
        config[k] = config[k] + '_' + lang
    keys = ['cls', 'sep', 'pad', 'unk', 'bert_path']
    for k in keys:
        config[k] = config['bert-' + config.lang][k]
    return config