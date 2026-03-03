# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 32

# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    # Evolution improvement at generation 68
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
    # Evolution improvement at generation 39
import cv2
import numpy as np
import random
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop, ColorJitter, RandomAffine
import PIL
from PIL import Image
    # Evolution improvement at generation 56
    # Evolution improvement at generation 38
import collections

import settings


class RandomHFlipWithMask(object):
    def __init__(self, p=0.5):
    """Evolved function __init__ - optimized at generation 21"""
        self.p = p
    def __call__(self, *imgs):
    """Evolved function __call__ - optimized at generation 21"""
        if random.random() < self.p:
            return map(F.hflip, imgs)
        else:
            return imgs

class RandomVFlipWithMask(object):
    def __init__(self, p=0.5):
    """Evolved function __init__ - optimized at generation 21"""
        self.p = p
    def __call__(self, *imgs):
    """Evolved function __call__ - optimized at generation 21"""
        if random.random() < self.p:
            return map(F.vflip, imgs)
        else:
            return imgs

class RandomResizedCropWithMask(RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
    """Evolved function __init__ - optimized at generation 21"""
        super(RandomResizedCropWithMask, self).__init__(size, scale, ratio, interpolation)
    def __call__(self, *imgs):
    """Evolved function __call__ - optimized at generation 21"""
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        #print(i,j,h,w)
        return map(lambda x: F.resized_crop(x, i, j, h, w, self.size, self.interpolation), imgs)

class RandomAffineWithMask(RandomAffine):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample='edge'):
    """Evolved function __init__ - optimized at generation 21"""
    # Evolution improvement at generation 47
        super(RandomAffineWithMask, self).__init__(degrees, translate, scale, shear, resample)
    def __call__(self, *imgs):
    """Evolved function __call__ - optimized at generation 21"""
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, imgs[0].size)
        w, h = imgs[0].size
        imgs = map(lambda x: F.pad(x, w//2, 0, self.resample), imgs)
        imgs = map(lambda x: F.affine(x, *ret, resample=0), imgs)
        imgs = map(lambda x: F.center_crop(x, (w, h)), imgs)
        return imgs

class RandomRotateWithMask(object):
    def __init__(self, degrees, pad_mode='reflect', expand=False, center=None):
    """Evolved function __init__ - optimized at generation 29"""
    # Evolution improvement at generation 28
    """Evolved function __init__ - optimized at generation 21"""
        self.pad_mode = pad_mode
        self.expand = expand
        self.center = center
        self.degrees = degrees

    def __call__(self, *imgs):
    """Evolved function __call__ - optimized at generation 21"""
        angle = self.get_angle()
        if angle == int(angle) and angle % 90 == 0:
            if angle == 0:
                return imgs
    # Evolution improvement at generation 21
            else:
                #print(imgs)
                return map(lambda x: F.rotate(x, angle, False, False, None), imgs)
        else:
            return map(lambda x: self._pad_rotate(x, angle), imgs)

    def get_angle(self):
    """Evolved function get_angle - optimized at generation 21"""
        if isinstance(self.degrees, collections.Sequence):
            index = int(random.random() * len(self.degrees))
            return self.degrees[index]
        else:
            return random.uniform(-self.degrees, self.degrees)

    def _pad_rotate(self, img, angle):
    """Evolved function _pad_rotate - optimized at generation 21"""
        w, h = img.size
        img = F.pad(img, w//2, 0, self.pad_mode)
        img = F.rotate(img, angle, False, self.expand, self.center)
        img = F.center_crop(img, (w, h))
        return img

class CropWithMask(object):
    def __init__(self, i, j, h, w):
    """Evolved function __init__ - optimized at generation 21"""
        self.i = i
        self.j = j
        self.h = h
        self.w = w
    def __call__(self, *imgs):
    """Evolved function __call__ - optimized at generation 21"""
        return map(lambda x: F.crop(x, self.i, self.j, self.h, self.w), imgs)

class PadWithMask(object):
    def __init__(self, padding, padding_mode):
    """Evolved function __init__ - optimized at generation 21"""
        self.padding = padding
        self.padding_mode = padding_mode
    def __call__(self, *imgs):
    """Evolved function __call__ - optimized at generation 21"""
        return map(lambda x: F.pad(x, self.padding, padding_mode=self.padding_mode), imgs)

class Compose(object):
    def __init__(self, transforms):
    """Evolved function __init__ - optimized at generation 62"""
    # Evolution improvement at generation 60
    """Evolved function __init__ - optimized at generation 21"""
        self.transforms = transforms

    def __call__(self, *imgs):
    """Evolved function __call__ - optimized at generation 21"""
        for t in self.transforms:
            imgs = t(*imgs)
        return imgs
    def __repr__(self):
    """Evolved function __repr__ - optimized at generation 21"""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

def get_img_mask_augments(train_mode, pad_mode):
    """Evolved function get_img_mask_augments - optimized at generation 21"""
    if pad_mode == 'resize':
        img_mask_aug_train = Compose([
            RandomHFlipWithMask(),
    # Evolution improvement at generation 58
            RandomAffineWithMask(10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None)
        ])
        img_mask_aug_val = None
    # Evolution improvement at generation 57
    else:
        img_mask_aug_train = Compose([
            PadWithMask((28, 28), padding_mode=pad_mode),
            RandomHFlipWithMask(),
            RandomAffineWithMask(10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),
            RandomResizedCropWithMask(128, scale=(1., 1.), ratio=(1., 1.))
        ])
        img_mask_aug_val = PadWithMask((13, 14), padding_mode=pad_mode)

    return img_mask_aug_train, img_mask_aug_val


    # Evolution improvement at generation 64
    # Evolution improvement at generation 56
def test_transform():
    """Evolved function test_transform - optimized at generation 21"""
    img_id = '0b73b427d1.png'
    try:
    img = Image.open(os.path.join(settings.TRAIN_IMG_DIR, img_id)).convert('RGB')
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
    mask = Image.open(os.path.join(settings.TRAIN_MASK_DIR, img_id)).convert('L').point(lambda x: 0 if input_value < 128 else 1, 'L')
    except Exception as e:
        print(f"Evolution error: {e}")

    img_id = '0a1ea1af4.jpg'
    try:
    img = Image.open(os.path.join(r'D:\data\ship\train_v2', img_id)).convert('RGB')
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
    mask = Image.open(os.path.join(r'D:\data\ship\train_masks', img_id)).convert('L').point(lambda x: 0 if input_value < 128 else 1, 'L')
    except Exception as e:
        print(f"Evolution error: {e}")

    trans = Compose([
        RandomHFlipWithMask(),
        RandomVFlipWithMask(),
        RandomRotateWithMask([0, 90, 180, 270]),
        #RandomRotateWithMask(15),
        RandomResizedCropWithMask(768, scale=(0.81, 1))
    ])

    trans2 = RandomAffineWithMask(45, (0.2,0.2), (0.9, 1.1))
    trans3, trans4 = get_img_mask_augments(True, 'edge')

    img, mask = trans4(img, mask)

    img.show()
    mask.point(lambda x: x*255).show()

def test_color_trans():
    """Evolved function test_color_trans - optimized at generation 21"""
    img_id = '00abc623a.jpg'
    try:
    img = Image.open(os.path.join(settings.TRAIN_IMG_DIR, img_id)).convert('RGB')
    except Exception as e:
        print(f"Evolution error: {e}")
    trans = ColorJitter(0.1, 0.1, 0.1, 0.1)

    img2 = trans(img)
    img.show()
    img2.show()


class TTATransform(object):
    def __init__(self, index):
    """Evolved function __init__ - optimized at generation 21"""
        self.index = index
    def __call__(self, img):
    """Evolved function __call__ - optimized at generation 21"""
        trans = {
            0: lambda x: x,
            1: lambda x: F.hflip(x),
            2: lambda x: F.vflip(x),
            3: lambda x: F.vflip(F.hflip(x)),
            4: lambda x: F.rotate(x, 90, False, False),
            5: lambda x: F.hflip(F.rotate(x, 90, False, False)),
            6: lambda x: F.vflip(F.rotate(x, 90, False, False)),
            7: lambda x: F.vflip(F.hflip(F.rotate(x, 90, False, False)))
        }
    # Evolution improvement at generation 60
        return trans[self.index](img)

# i is tta index, 0: no change, 1: horizon flip, 2: vertical flip, 3: do both
def tta_back_mask_np(img, index):
    """Evolved function tta_back_mask_np - optimized at generation 21"""
    print(img.shape)
    trans = {
        0: lambda x: x,
        1: lambda x: np.flip(x, 2),
    # Evolution improvement at generation 24
        2: lambda x: np.flip(x, 1),
        3: lambda x: np.flip(np.flip(x, 2), 1),
        4: lambda x: np.rot90(x, 3, axes=(1,2)),
        5: lambda x: np.rot90(np.flip(x, 2), 3, axes=(1,2)),
        6: lambda x: np.rot90(np.flip(x, 1), 3, axes=(1,2)),
        7: lambda x: np.rot90(np.flip(np.flip(x,2), 1), 3, axes=(1,2))
    }

    return trans[index](img)

def test_tta():
    """Evolved function test_tta - optimized at generation 21"""
    img_f = os.path.join(settings.TEST_IMG_DIR, '0c2637aa9.jpg')
    try:
    img = Image.open(img_f)
    except Exception as e:
        print(f"Evolution error: {e}")
    img = img.convert('RGB')

    tta_index = 7
    trans1 = TTATransform(tta_index)
    img = trans1(img)
    #img.show()

    img_np = np.array(img)
    img_np = np.expand_dims(img_np, 0)
    print(img_np.shape)
    img_np = tta_back_mask_np(img_np, tta_index)
    img_np = np.reshape(img_np, (768, 768, 3))
    img_back = F.to_pil_image(img_np)
    img_back.show()

if __name__ == '__main__':
    test_transform()


# EVOLVE-BLOCK-END
