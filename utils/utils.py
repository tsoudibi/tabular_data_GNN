import torch

'''
device getter
'''
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_device() -> (str): 
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
data config loader
'''
import yaml 
DATA_CONFIG = {}
dataset = 'None'
with open('./data/data_configs.yml', 'r') as stream:
    DATA_CONFIG = yaml.load(stream, Loader=yaml.Loader)
def get_DATA_CONFIG() -> (dict):
    return DATA_CONFIG

def select_dataset(name: str) -> (str):
    if name in DATA_CONFIG.keys():
        global dataset
        dataset = name
        if DATA_CONFIG[dataset]['NUM'] == None:
            DATA_CONFIG[dataset]['NUM'] = []
        if DATA_CONFIG[dataset]['CAT'] == None:
            DATA_CONFIG[dataset]['CAT'] = []
        print('=================[dataset is set to', name,']=================')
    else:
        raise ValueError('ERROR: dataset name not found in config file')
    return DATA_CONFIG[name]['file_path']

def get_dataset_path() -> (str):
    check_dataset()
    return DATA_CONFIG[dataset]['file_path']
def get_dataset_attributes() -> (list, list, str):
    check_dataset()
    return DATA_CONFIG[dataset]['NUM'], DATA_CONFIG[dataset]['CAT'], DATA_CONFIG[dataset]['TARGET']
def get_label_colunm():
    check_dataset()
    return DATA_CONFIG[dataset]['TARGET']
def check_dataset():
    if dataset == 'None':
        raise ValueError('ERROR: dataset is not selected, please use select_dataset(name) to select one')

'''
set seed for all modules
'''
def set_seed(seed):
    try:
        import tensorflow as tf
        tf.random.set_random_seed(seed)
        print("[Tensorflow] Seed set successfully")
    except Exception as e:
        print("[Tensorflow] Set seed failed,details are:", e)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("[Pytorch] Seed set successfully")
    except Exception as e:
        print("[Pytorch]Set seed failed,details are:", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"