from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF


class PatchSampler(ABC):
    def __init__(self, patch_size=(16, 16), seed=1000000009):
        self.patch_size = patch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_state(self):
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def __call__(self, img):
        raise NotImplemented()


class SimplePatchSampler(PatchSampler):
    def _get_coordinates(self, img_size, img):
        raise NotImplemented()

    def __call__(self, img):
        assert len(img.shape) == 3 and img.shape[0] == 3
        img_size = img.shape[1:]

        y_x_s = self._get_coordinates(img_size, img)

        patches = []
        coords = []
        for y, x, s in y_x_s:
            patch = img[:, y:y + s, x:x + s]
            if patch.shape[-1] != 16 or patch.shape[-2] != 16:
                patch = TF.resize(patch, [16, 16], antialias=False)
            patches.append(patch)
            coords.append([y, x, y + s, x + s])

        return torch.stack(patches, dim=0), torch.LongTensor(coords)


class GridSampler(SimplePatchSampler):
    def __init__(self, patch_size=(16, 16), seed=1000000009):
        assert patch_size[0] == patch_size[1]
        super().__init__(patch_size, seed)

    def _get_coordinates(self, img_size, img):
        return [(i, j, self.patch_size[0]) for i in range(0, img_size[0], self.patch_size[0]) for j in
                range(0, img_size[1], self.patch_size[0])]


class RandomMultiscaleSampler(SimplePatchSampler):
    def __init__(self, crop_sizes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.crop_sizes = np.asarray(crop_sizes)

    def _get_coordinates(self, img_size, img):
        sizes = self.crop_sizes
        ys = self.rng.integers(0, img_size[0] - sizes)
        xs = self.rng.integers(0, img_size[1] - sizes)
        return zip(ys, xs, sizes)


class RandomUniformSampler(SimplePatchSampler):
    def __init__(self, random_patches, min_patch_size=8, max_patch_size=48, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_patches = random_patches
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

    def _get_coordinates(self, img_size, img):
        sizes = (self.rng.integers(self.min_patch_size,
                                   self.max_patch_size,
                                   size=self.random_patches)).astype(np.int64)
        ys = self.rng.integers(0, img_size[0] - sizes)
        xs = self.rng.integers(0, img_size[1] - sizes)
        return zip(ys, xs, sizes)


class RandomDelegatedSampler(PatchSampler):

    def __init__(self, samplers: List[Tuple[PatchSampler, float]], patch_size=(16, 16), seed=1000000009):
        super().__init__(patch_size, seed)

        self.samplers, self.proba = zip(*samplers)
        for sampler in self.samplers:
            sampler.rng = self.rng
        self.proba = np.asarray(self.proba)
        self.proba = self.proba / self.proba.sum()

    def __call__(self, img):
        sampler = self.rng.choice(self.samplers, p=self.proba)
        return sampler(img)
