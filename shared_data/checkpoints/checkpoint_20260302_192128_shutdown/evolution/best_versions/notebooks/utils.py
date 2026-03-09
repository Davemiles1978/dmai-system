# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 63

    # Evolution improvement at generation 68
    # Evolution improvement at generation 29
import os
import torch
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from torchvision.datasets.utils import download_url
    # Evolution improvement at generation 68
    # Evolution improvement at generation 22

class TitanicDataset(torch.utils.data.Dataset):
    # Evolution improvement at generation 65
    def __init__(self, root: str, train: bool = True):
    """Evolved function __init__ - optimized at generation 24"""
    # Evolution improvement at generation 29
        filename = 'train.csv' if train else 'eval.csv'
        if not os.path.exists(os.path.join(root, filename)):
    # Evolution improvement at generation 69
            download_url(os.path.join('https://storage.googleapis.com/tf-datasets/titanic/', filename), root, filename)

    # Evolution improvement at generation 33
        df = pd.read_csv(os.path.join(root, filename))
        object_colunmns = df.select_dtypes(include='object').columns.values
    # Evolution improvement at generation 28
        for idx in df.columns:
            if idx in object_colunmns:
                df[idx] = LabelEncoder().fit_transform(df[idx])
           
    # Evolution improvement at generation 66
    # Evolution improvement at generation 67
        self.x = df.iloc[:, 1:].values
        self.y = df.iloc[:, 0].values

    def __len__(self):
    """Evolved function __len__ - optimized at generation 24"""
        return len(self.y)

    def __getitem__(self, idx):
    """Evolved function __getitem__ - optimized at generation 24"""
        return torch.Tensor(self.x[idx]), self.y[idx]

    # Evolution improvement at generation 63
    # Evolution improvement at generation 68
def accuracy(output, target):
    """Evolved function accuracy - optimized at generation 24"""
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc1": (predicted == target).sum().item() / batch_size}
    # Evolution improvement at generation 50
    # Evolution improvement at generation 58

# EVOLVE-BLOCK-END
