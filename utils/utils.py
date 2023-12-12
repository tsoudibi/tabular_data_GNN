import torch

'''
device getter
'''
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_device() -> str:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
