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


class InteractiveSampler:
    def __init__(self, images: torch.Tensor, native_patch_size=(16, 16)):
        # B x C x H x W
        assert len(images.shape) == 4 and images.shape[1] == 3
        self._images = images
        self._batch_size = images.shape[0]
        self._native_patch_size = native_patch_size

        self._patches = None
        self._coords = None
        self.initialize()

    def sample_multi_relative(self, new_crops, grid_size=2):
        # B x (y, x, s)
        new_crops = new_crops.detach().clone()
        new_crops[..., 2] *= min(self._images.shape[-1], self._images.shape[-1])
        new_crops[..., 0] = new_crops[..., 0] * (self._images.shape[-2] - new_crops[..., 2])
        new_crops[..., 1] = new_crops[..., 1] * (self._images.shape[-1] - new_crops[..., 2])
        new_crops = torch.floor(new_crops)

        patch_size = torch.floor(new_crops[..., 2] / grid_size)
        last_patch_size = new_crops[..., 2] - patch_size * (grid_size - 1)

        new_crops = new_crops.unsqueeze(1).repeat((1, grid_size * grid_size, 1))

        for y in range(grid_size):
            for x in range(grid_size):
                new_crops[:, y * grid_size + x, 0] += y * patch_size
                new_crops[:, y * grid_size + x, 1] += x * patch_size
                if y == grid_size - 1 or x == grid_size - 1:
                    new_crops[:, y * grid_size + x, 2] = last_patch_size
                else:
                    new_crops[:, y * grid_size + x, 2] = patch_size

        new_crops = new_crops.to(torch.long)
        return self.sample(new_crops)

    def sample(self, new_crops: torch.Tensor):

        assert new_crops.shape[0] == self._batch_size
        assert new_crops.shape[2] == 3
        assert new_crops.dtype == torch.long

        num_patches = new_crops.shape[1]

        new_patches = []
        new_coords = []

        # rework for batched operation in future
        for idx in range(self._batch_size):
            for patch in range(num_patches):
                y, x, s = new_crops[idx, patch]
                patch = self._images[idx, :, y:y + s, x:x + s]

                if patch.shape[-1] != self._native_patch_size[0] or patch.shape[-2] != self._native_patch_size[1]:
                    patch = TF.resize(patch, list(self._native_patch_size), antialias=False)
                new_patches.append(patch)
                new_coords.append([y, x, y + s, x + s])

        new_patches = torch.stack(new_patches, dim=0).reshape(self._batch_size, num_patches, 3,
                                                              self._native_patch_size[0],
                                                              self._native_patch_size[1])
        new_coords = torch.tensor(new_coords, device=self._images.device, dtype=torch.long).reshape(self._batch_size,
                                                                                                    num_patches, 4)

        self._patches = torch.cat([self._patches, new_patches], dim=1)
        self._coords = torch.cat([self._coords, new_coords], dim=1)

        return self._patches, self._coords

    def initialize(self):
        self._patches = torch.zeros((self._batch_size, 0, 3, self._native_patch_size[0], self._native_patch_size[1]),
                                    dtype=self._images.dtype, device=self._images.device)
        self._coords = torch.zeros((self._batch_size, 0, 4), dtype=torch.long, device=self._images.device)
        self.do_grid(grid_size=4)  # 4 glimpses, change in future

    @property
    def patches(self):
        return self._patches

    @property
    def coords(self):
        return self._coords

    @property
    def num_patches(self):
        return self._coords.shape[1]

    def do_grid(self, grid_size=4):
        samples = []
        start_size = self._images.shape[-1] // grid_size
        for y in range(0, 224, start_size):
            for x in range(0, 224, start_size):
                samples.append([y, x, start_size])
        samples = torch.LongTensor(samples).unsqueeze(0)
        samples = samples.repeat((self._batch_size, 1, 1))
        self.sample(samples)
