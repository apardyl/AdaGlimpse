# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""

import random

import torch
from PIL import ImageFilter, ImageOps
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision import transforms


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GrayScale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def three_augment(img_size, src=False, color_jitter=0.3):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    scale = (0.08, 1.0)
    interpolation = 'bicubic'
    if src:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

    secondary_tfl = [transforms.RandomChoice([GrayScale(p=1.0),
                                              Solarization(p=1.0),
                                              GaussianBlur(p=1.0)])]

    if color_jitter is not None and not color_jitter == 0:
        secondary_tfl.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))
    final_tfl = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ]
    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
