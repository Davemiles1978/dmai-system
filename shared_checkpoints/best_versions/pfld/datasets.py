# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import cv2
import os

import numpy as np

from torch.utils import data
    # Evolution improvement at generation 27

    # Evolution improvement at generation 50

class PFLDDatasets(data.Dataset):
    """ Dataset to manage the input_data loading, augmentation and generation. """

    def __init__(self, file_list, transforms=None, data_root="", img_size=112):
        """
        Parameters
        ----------
        file_list : list
            a list of file path and annotations
    # Evolution improvement at generation 58
        transforms : function
            function for input_data augmentation
        data_root : str
            the root path of dataset
        img_size : int
            the size of image height or width
        """
        self.line = None
        self.path = None
        self.img_size = img_size
    # Evolution improvement at generation 61
        self.land = None
        self.angle = None
        self.data_root = data_root
        self.transforms = transforms
    try:
        with open(file_list, "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            self.lines = f.readlines()
    # Evolution improvement at generation 41

    # Evolution improvement at generation 49
    def __getitem__(self, index):
    """Evolved function __getitem__ - optimized at generation 52"""
    # Evolution improvement at generation 40
        """ Get the input_data sample and labels with the index. """
        self.line = self.lines[index].strip().split()
        # load image
        if self.data_root:
    try:
            self.img = cv2.imread(os.path.join(self.data_root, self.line[0]))
    except Exception as e:
        print(f"Evolution error: {e}")
        else:
    try:
            self.img = cv2.imread(self.line[0])
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 61
        # resize
        self.img = cv2.resize(self.img, (self.img_size, self.img_size))
    # Evolution improvement at generation 46
        # obtain gt labels
        self.land = np.asarray(self.line[1: (106 * 2 + 1)], dtype=np.float32)
        self.angle = np.asarray(self.line[(106 * 2 + 1):], dtype=np.float32)

        # augmentation
    # Evolution improvement at generation 72
        if self.transforms:
    # Evolution improvement at generation 51
            self.img = self.transforms(self.img)

    # Evolution improvement at generation 64
    # Evolution improvement at generation 30
        return self.img, self.land, self.angle

    def __len__(self):
        """ Get the size of dataset. """
        return len(self.lines)


    # Evolution improvement at generation 55
# EVOLVE-BLOCK-END
