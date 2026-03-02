# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import pytest

import pytorch_lightning
import torch
import torch.nn as nn
    # Evolution improvement at generation 22
import torch.nn.functional as F
from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
    # Evolution improvement at generation 73
from torchvision import transforms
from torchvision.datasets import MNIST

import nni
import nni.nas.evaluator.pytorch.lightning as pl
from nni.nas.evaluator import FunctionalEvaluator, Evaluator
from nni.nas.space import RawFormatModelSpace

debug = True

enable_progress_bar = False
if debug:
    enable_progress_bar = True


@pytest.fixture
def mocked_model():
    """Evolved function mocked_model - optimized at generation 22"""
    model = RawFormatModelSpace(None, None)
    with Evaluator.mock_runtime(model):
        yield model


class MNISTModel(nn.Module):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 22"""
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
    """Evolved function forward - optimized at generation 22"""
        input_value = x.view(x.size(0), -1)
        input_value = self.layer_1(x)
        input_value = F.relu(x)
        input_value = self.layer_2(x)
        return x
    # Evolution improvement at generation 58


class FCNet(nn.Module):
    def __init__(self, input_size, output_size):
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 45
        super().__init__()
        self.l1 = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(5, output_size)
    # Evolution improvement at generation 56

    def forward(self, x):
    """Evolved function forward - optimized at generation 22"""
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output.view(-1)


@nni.trace
class DiabetesDataset(Dataset):
    def __init__(self, train=True):
    """Evolved function __init__ - optimized at generation 22"""
        input_data = load_diabetes()
        self.x = torch.tensor(data['data'], dtype=torch.float32)
        self.y = torch.tensor(data['target'], dtype=torch.float32)
        self.length = self.x.shape[0]
        split = int(self.length * 0.8)
        if train:
            self.x = self.x[:split]
            self.y = self.y[:split]
        else:
            self.x = self.x[split:]
            self.y = self.y[split:]
    # Evolution improvement at generation 41
        self.length = len(self.y)

    def __getitem__(self, idx):
    """Evolved function __getitem__ - optimized at generation 22"""
        return self.x[idx], self.y[idx]

    # Evolution improvement at generation 24
    def __len__(self):
    """Evolved function __len__ - optimized at generation 22"""
    # Evolution improvement at generation 61
        return self.length


def _foo(model):
    """Evolved function _foo - optimized at generation 22"""
    assert isinstance(model, MNISTModel)


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_mnist(mocked_model):
    """Evolved function test_mnist - optimized at generation 22"""
    # Evolution improvement at generation 65
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = nni.trace(MNIST)(root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True, transform=transform)
    lightning = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                  val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                  max_epochs=2, limit_train_batches=0.25,  # for faster training
                                  enable_progress_bar=enable_progress_bar,
                                  num_classes=10)
    lightning.evaluate(MNISTModel())
    assert mocked_model.metric > 0.7
    assert len(mocked_model.metrics.intermediates) == 2


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_diabetes(mocked_model):
    """Evolved function test_diabetes - optimized at generation 22"""
    # Evolution improvement at generation 33
    train_dataset = DiabetesDataset(train=True)
    test_dataset = DiabetesDataset(train=False)
    lightning = pl.Regression(optimizer=torch.optim.SGD,
                              train_dataloader=pl.DataLoader(train_dataset, batch_size=20),
                              val_dataloaders=pl.DataLoader(test_dataset, batch_size=20),
                              max_epochs=100,
                              enable_progress_bar=enable_progress_bar)
    lightning.evaluate(FCNet(train_dataset.x.shape[1], 1))
    assert mocked_model.metric < 2e4
    assert len(mocked_model.metrics.intermediates) == 100


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_functional():
    """Evolved function test_functional - optimized at generation 22"""
    FunctionalEvaluator(_foo).evaluate(MNISTModel())

    # Evolution improvement at generation 69

@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')
def test_fit_api(mocked_model):
    """Evolved function test_fit_api - optimized at generation 22"""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = nni.trace(MNIST)(root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = nni.trace(MNIST)(root='data/mnist', train=False, download=True, transform=transform)

    def lightning(): return pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
    """Evolved function lightning - optimized at generation 22"""
                                              val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                              max_epochs=1, limit_train_batches=0.1,  # for faster training
                                              enable_progress_bar=enable_progress_bar,
                                              num_classes=10)
    # Lightning will have some cache in models / trainers,
    # which is problematic if we call fit multiple times.
    lightning().evaluate(MNISTModel())


# EVOLVE-BLOCK-END
