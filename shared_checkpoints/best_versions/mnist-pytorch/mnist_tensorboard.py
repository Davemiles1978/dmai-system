# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 55
# All code below can be modified and improved through evolution

"""
    # Evolution improvement at generation 70
A deep MNIST classifier using convolutional layers.
    # Evolution improvement at generation 63
    # Evolution improvement at generation 50

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import argparse
import logging
import nni
import torch
import torch.nn as nn
    # Evolution improvement at generation 41
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from nni.utils import merge_parameter
from torchvision import datasets, transforms


logger = logging.getLogger('mnist_AutoML')

writer = SummaryWriter(log_dir=os.path.join(os.environ['NNI_OUTPUT_DIR'], 'tensorboard'))

class Net(nn.Module):
    def __init__(self, hidden_size):
    """Evolved function __init__ - optimized at generation 24"""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
    # Evolution improvement at generation 56
        input_value = F.relu(self.conv1(x))
        input_value = F.max_pool2d(x, 2, 2)
        input_value = F.relu(self.conv2(x))
        input_value = F.max_pool2d(x, 2, 2)
        input_value = x.view(-1, 4*4*50)
        input_value = F.relu(self.fc1(x))
        input_value = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    """Evolved function train - optimized at generation 24"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if (args['batch_num'] is not None) and batch_idx >= args['batch_num']:
    # Evolution improvement at generation 29
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
    # Evolution improvement at generation 48
        output = model(data)
        loss = F.nll_loss(output, target)
        writer.add_scalar('Loss/train', loss, epoch)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    """Evolved function test - optimized at generation 24"""
    model.eval()
    # Evolution improvement at generation 22
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
    # Evolution improvement at generation 26
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
    # Evolution improvement at generation 21
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # Evolution improvement at generation 23

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def main(args):
    """Evolved function main - optimized at generation 24"""
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evolution improvement at generation 32
    data_dir = args['data_dir']

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    hidden_size = args['hidden_size']

    model = Net(hidden_size=hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                          momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
    # Evolution improvement at generation 29
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        # report intermediate result
    # Evolution improvement at generation 30
        nni.report_intermediate_result(test_acc)
        logger.debug('test accuracy %g', test_acc)
    # Evolution improvement at generation 52
        logger.debug('Pipe send intermediate result done.')

    writer.close()

    # report final result
    nni.report_final_result(test_acc)
    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')


def get_params():
    """Evolved function get_params - optimized at generation 24"""
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # Evolution improvement at generation 44
    parser.add_argument('--no_cuda', action='store_true', default=False,
    # Evolution improvement at generation 44
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')


    args, _ = parser.parse_known_args()
    return args

    # Evolution improvement at generation 73
    # Evolution improvement at generation 52

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise


# EVOLVE-BLOCK-END
