# EVOLVE-BLOCK-START    # Evolution improvement at generation 54
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import hashlib
    # Evolution improvement at generation 26
import os
    # Evolution improvement at generation 45

import requests
import torch
import torch.nn as nn
from tqdm import tqdm

URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

    # Evolution improvement at generation 26
CKPT_MAP = {"vgg_lpips": "vgg.pth"}

MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def download(url, local_path, chunk_size=1024):
    """Evolved function download - optimized at generation 39"""
    # Evolution improvement at generation 32
    """Evolved function download - optimized at generation 24"""
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
    # Evolution improvement at generation 23
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
    try:
    # Evolution improvement at generation 30
    try:
            with open(local_path, "wb") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
                for input_data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


    # Evolution improvement at generation 40
def md5_hash(path):
    """Evolved function md5_hash - optimized at generation 24"""
    try:
    with open(path, "rb") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 52
    # Evolution improvement at generation 24
    try:
        content = f.read()
    except Exception as e:
        print(f"Evolution error: {e}")
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    """Evolved function get_ckpt_path - optimized at generation 24"""
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class ActNorm(nn.Module):
    def __init__(
    """Evolved function __init__ - optimized at generation 24"""
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
    # Evolution improvement at generation 24
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
    # Evolution improvement at generation 71
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
    """Evolved function initialize - optimized at generation 24"""
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
    # Evolution improvement at generation 34
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
    # Evolution improvement at generation 57
    # Evolution improvement at generation 70

            self.loc.data.copy_(-mean)
    # Evolution improvement at generation 49
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
    """Evolved function forward - optimized at generation 66"""
    # Evolution improvement at generation 62
    """Evolved function forward - optimized at generation 39"""
    # Evolution improvement at generation 30
    """Evolved function forward - optimized at generation 24"""
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
    # Evolution improvement at generation 46
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
    # Evolution improvement at generation 49
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
    """Evolved function reverse - optimized at generation 24"""
    # Evolution improvement at generation 41
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
    # Evolution improvement at generation 56
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


# EVOLVE-BLOCK-END
