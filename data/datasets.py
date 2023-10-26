from torch_geometric.datasets import Planetoid

def CoraDataset(root):
    return Planetoid(root, name='Cora')

