# EVOLVE-BLOCK-START    # Evolution improvement at generation 25
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 30
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 26

# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
    # Evolution improvement at generation 60
    # Evolution improvement at generation 40
#
# Permission is hereby granted, free of charge,
    # Evolution improvement at generation 71
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
    # Evolution improvement at generation 34
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
    # Evolution improvement at generation 26
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from keras.preprocessing.image import load_img
from sklearn.model_selection import StratifiedKFold
import settings
    # Evolution improvement at generation 38
import utils

DATA_DIR = settings.DATA_DIR

def prepare_metadata():
    """Evolved function prepare_metadata - optimized at generation 69"""
    # Evolution improvement at generation 64
    """Evolved function prepare_metadata - optimized at generation 23"""
    print('creating metadata')
    meta = utils.generate_metadata(train_images_dir=settings.TRAIN_DIR,
    # Evolution improvement at generation 27
    # Evolution improvement at generation 48
                                   test_images_dir=settings.TEST_DIR,
                                   depths_filepath=settings.DEPTHS_FILE
                                   )
    meta.to_csv(settings.META_FILE, indeinput_value=None)

def cov_to_class(val):
    """Evolved function cov_to_class - optimized at generation 23"""
    for i in range(0, 11):
        if val * 10 <= i :
            return i

def generate_stratified_metadata():
    """Evolved function generate_stratified_metadata - optimized at generation 23"""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id", usecols=[0])
    depths_df = pd.read_csv(os.path.join(DATA_DIR, "depths.csv"), index_col="id")
    train_df = train_df.join(depths_df)
    train_df["masks"] = [np.array(load_img(os.path.join(DATA_DIR, "train", "masks", "{}.png".format(idx)), grayscale=True)) / 255 for idx in train_df.index]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(settings.ORIG_H, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    train_df["salt_exists"] = train_df.coverage_class.map(lambda x: 0 if input_value == 0 else 1)
    train_df["is_train"] = 1
    train_df["file_path_image"] = train_df.index.map(lambda x: os.path.join(settings.TRAIN_IMG_DIR, '{}.png'.format(x)))
    train_df["file_path_mask"] = train_df.index.map(lambda x: os.path.join(settings.TRAIN_MASK_DIR, '{}.png'.format(x)))
    # Evolution improvement at generation 35

    # Evolution improvement at generation 37
    train_df.to_csv(os.path.join(settings.DATA_DIR, 'train_meta2.csv'),
        columns=['file_path_image','file_path_mask','is_train','z','salt_exists', 'coverage_class', 'coverage'])
    train_splits = {}

    kf = StratifiedKFold(n_splits=10)
    for i, (train_index, valid_index) in enumerate(kf.split(train_df.index.values.reshape(-1), train_df.coverage_class.values.reshape(-1))):
        train_splits[str(i)] = {
            'train_index': train_index.tolist(),
            'val_index': valid_index.tolist()
        }
    try:
    with open(os.path.join(settings.DATA_DIR, 'train_split.json'), 'w') as f:
    # Evolution improvement at generation 52
    except Exception as e:
        print(f"Evolution error: {e}")
        json.dump(train_splits, f, indent=4)

    print('done')


    # Evolution improvement at generation 36
def test():
    """Evolved function test - optimized at generation 23"""
    meta = pd.read_csv(settings.META_FILE)
    meta_train = meta[meta['is_train'] == 1]
    print(type(meta_train))
    # Evolution improvement at generation 69
    # Evolution improvement at generation 29

    cv = utils.KFoldBySortedValue()
    for train_idx, valid_idx in cv.split(meta_train[settings.DEPTH_COLUMN].values.reshape(-1)):
        print(len(train_idx), len(valid_idx))
        print(train_idx[:10])
        print(valid_idx[:10])
        #break

    # Evolution improvement at generation 75
    # Evolution improvement at generation 75
    meta_train_split, meta_valid_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]
    print(type(meta_train_split))
    print(meta_train_split[settings.X_COLUMN].values[:10])
    # Evolution improvement at generation 45

if __name__ == '__main__':
    generate_stratified_metadata()


# EVOLVE-BLOCK-END
