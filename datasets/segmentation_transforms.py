import numpy as np
import torch
import torchvision.transforms.functional as F

from torchvision.transforms.transforms import RandomAffine, RandomHorizontalFlip, ColorJitter, RandomCrop, Resize, RandomResizedCrop
from torchvision.transforms import InterpolationMode

from datasets.utils import IMAGENET_MEAN, IMAGENET_STD


class MaskCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomMaskHorizontalFlip(RandomHorizontalFlip):
    """Horizontally flip the given sample (image, and mask) randomly with a given probability"""

    def __init__(self, p=0.5):
        super().__init__(p)
        self.p = p

    def forward(self, image, target):
        if torch.rand(1) < self.p:
            return F.hflip(image), F.hflip(target)
        return image, target


class ColorMaskJitter(ColorJitter):
    """Randomly change the brightness, contrast and saturation of an image, don't change the mask."""

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        super().__init__(brightness, contrast, saturation, hue)

    def forward(self, image, target):
        image = super().forward(image)
        return image, target


class RandomMaskAffine(RandomAffine):
    """ Random affine transformation of the image and mask"""

    def __init__(self, degrees, translate=None, scale=None, shear=None, interpolation=InterpolationMode.BICUBIC, fill=0):
        super().__init__(degrees, translate, scale, shear, interpolation, fill)

    def forward(self, image, target):
        img_size = image.size
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        return F.affine(image, *ret, interpolation=self.interpolation, fill=self.fill), \
               F.affine(target, *ret, interpolation=InterpolationMode.NEAREST, fill=self.fill)


class RandomMaskCrop(RandomCrop):
    """ Random affine transformation of the image and mask"""

    def __init__(self, size, padding=None, pad_if_needed=True, fill=0, padding_mode="constant"):
        super().__init__(size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            target = F.pad(target, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            target = F.pad(target, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            target = F.pad(target, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(target, i, j, h, w)


class MaskResize(Resize):
    def __init__(self, size, interpolation=InterpolationMode.BICUBIC):
        super().__init__(size, interpolation)

    def __call__(self, image, target):
        image = F.resize(image, self.size, self.interpolation, self.max_size, self.antialias)
        target = F.resize(target, self.size, InterpolationMode.NEAREST, self.max_size, self.antialias)
        return image, target


class RandomMaskResizedCrop(RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)

    def __call__(self, image, target):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        return F.resized_crop(image, i, j, h, w, self.size, self.interpolation), F.resized_crop(target, i, j, h, w, self.size, InterpolationMode.NEAREST)


class MaskNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class MaskToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


def get_aug_seg_transforms(img_size):
    return MaskCompose([
        RandomMaskHorizontalFlip(0.5),
        RandomMaskAffine(10, (0.05, 0.05), scale=(1.0, 1.0), shear=(5, 5), interpolation=InterpolationMode.BICUBIC),
        ColorMaskJitter(0.3, 0.3, 0.05, 0.05),
        # RandomMaskCrop((IMG_SIZE[1], IMG_SIZE[0]), pad_if_needed=True),
        MaskResize(img_size),
        # RandomMaskResizedCrop(img_size, scale=(0.08, 1.0), ratio=(1.0, 3.0)),
        MaskToTensor(),
        MaskNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_seg_transforms(img_size):
    return MaskCompose([
        MaskResize(img_size),
        MaskToTensor(),
        MaskNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

