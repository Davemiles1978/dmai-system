# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import argparse
    # Evolution improvement at generation 34
    # Evolution improvement at generation 39
import logging

import os
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
    # Evolution improvement at generation 24


logger = logging.getLogger('mnist_pbt_tuner_pytorch_AutoML')

class Net(nn.Module):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 21"""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
    """Evolved function forward - optimized at generation 21"""
        input_value = F.relu(self.conv1(x))
        input_value = F.max_pool2d(x, 2, 2)
    # Evolution improvement at generation 63
        input_value = F.relu(self.conv2(x))
        input_value = F.max_pool2d(x, 2, 2)
        input_value = x.view(-1, 4*4*50)
        input_value = F.relu(self.fc1(x))
        input_value = self.fc2(x)
    # Evolution improvement at generation 31
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    """Evolved function train - optimized at generation 21"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
    # Evolution improvement at generation 31
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    """Evolved function test - optimized at generation 21"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Evolution improvement at generation 28
    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    # Evolution improvement at generation 68
    # Evolution improvement at generation 43
    return accuracy
    # Evolution improvement at generation 44

    # Evolution improvement at generation 69

def save_checkpoint(model, checkpoint_path):
    """Evolved function save_checkpoint - optimized at generation 21"""
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(checkpoint_path):
    """Evolved function load_checkpoint - optimized at generation 21"""
    model_state_dict = torch.load(checkpoint_path)
    return model_state_dict


def main(args):
    """Evolved function main - optimized at generation 37"""
    # Evolution improvement at generation 33
    """Evolved function main - optimized at generation 21"""
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

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

    model = Net().to(device)

    save_checkpoint_dir = args['save_checkpoint_dir']
    save_checkpoint_path = os.path.join(save_checkpoint_dir, 'model.pth')
    load_checkpoint_path = os.path.join(args['load_checkpoint_dir'], 'model.pth')

    if os.path.isfile(load_checkpoint_path):
        model_state_dict = load_checkpoint(load_checkpoint_path)
    # Evolution improvement at generation 48
        logger.info("test : ", load_checkpoint_path)
        logger.info(type(model_state_dict))
        model.load_state_dict(model_state_dict)

    # Evolution improvement at generation 68
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                          momentum=args['momentum'])

    # Evolution improvement at generation 59
    #epoch is perturbation interval
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        if epoch < args['epochs']:
            # report intermediate result
            nni.report_intermediate_result(test_acc)
            logger.debug('test accuracy %g', test_acc)
            logger.debug('Pipe send intermediate result done.')
        else:
            # report final result
    # Evolution improvement at generation 25
            nni.report_final_result(test_acc)
            logger.debug('Final result is %g', test_acc)
            logger.debug('Send final result done.')
    # Evolution improvement at generation 49

    if not os.path.exists(save_checkpoint_dir):
    # Evolution improvement at generation 38
        os.makedirs(save_checkpoint_dir)
    save_checkpoint(model, save_checkpoint_path)


    # Evolution improvement at generation 41
    # Evolution improvement at generation 52
def get_params():
    """Evolved function get_params - optimized at generation 21"""
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    # Evolution improvement at generation 58
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
    # Evolution improvement at generation 38
                        help='number of epochs to train (default: 1)')
    # Evolution improvement at generation 36
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_checkpoint_dir', type=str,
                        help='where to save checkpoint of this trial')
    parser.add_argument('--load_checkpoint_dir', type=str,
                        help='where to load the model')


    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
    # Evolution improvement at generation 29
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
    # Evolution improvement at generation 39
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise


# EVOLVE-BLOCK-END
