from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize

from architectures.rl.shared_memory import SharedMemory


class InteractiveStatelessSampler:
    def __init__(self, glimpse_grid_size: int, max_glimpses: int, native_patch_size: Tuple[int, int],
                 max_size_ratio=1):
        self._native_patch_size = native_patch_size
        self._glimpse_grid_size = glimpse_grid_size
        self._patches_per_glimpse = glimpse_grid_size ** 2
        self._max_glimpses = max_glimpses
        self._max_patches = max_glimpses * self._patches_per_glimpse
        self._max_size_ratio = max_size_ratio

    def _calculate_crops(self, new_crops: Tensor, image_shape: torch.Size) -> Tensor:
        max_crop_size = (
                min(image_shape[-2], image_shape[-1]) * self._max_size_ratio)
        min_crop_size = min(self._native_patch_size) * self._glimpse_grid_size

        new_crops[..., 2] = new_crops[..., 2] * (max_crop_size - min_crop_size) + min_crop_size
        new_crops[..., 0] = new_crops[..., 0] * (image_shape[-2] - new_crops[..., 2])
        new_crops[..., 1] = new_crops[..., 1] * (image_shape[-1] - new_crops[..., 2])
        new_crops = torch.floor(new_crops)

        patch_size = torch.floor(new_crops[..., 2] / self._glimpse_grid_size)
        last_patch_size = new_crops[..., 2] - patch_size * (self._glimpse_grid_size - 1)

        new_crops = new_crops.unsqueeze(1).repeat((1, self._patches_per_glimpse, 1))

        for y in range(self._glimpse_grid_size):
            for x in range(self._glimpse_grid_size):
                new_crops[:, y * self._glimpse_grid_size + x, 0] += y * patch_size
                new_crops[:, y * self._glimpse_grid_size + x, 1] += x * patch_size
                if y == self._glimpse_grid_size - 1 or x == self._glimpse_grid_size - 1:
                    new_crops[:, y * self._glimpse_grid_size + x, 2] = last_patch_size
                else:
                    new_crops[:, y * self._glimpse_grid_size + x, 2] = patch_size

        new_crops = new_crops.to(torch.long)
        return new_crops

    def _crop_and_resize(self, new_crops: torch.Tensor, images_cpu):
        patches = []
        coords = []

        for b_idx in range(new_crops.shape[0]):  # batch size
            for p_idx in range(new_crops.shape[1]):  # crops
                y, x, s = new_crops[b_idx, p_idx]
                patch = images_cpu[b_idx, :, y:y + s, x:x + s]

                if patch.shape[-1] != self._native_patch_size[0] or patch.shape[-2] != self._native_patch_size[1]:
                    patch = resize(
                        patch, list(self._native_patch_size),
                        antialias=False,
                        interpolation=InterpolationMode.BILINEAR
                    )

                patches.append(patch)
                coord = torch.tensor([y, x, y + s, x + s], dtype=torch.long)
                coords.append(coord)

        patches = torch.stack(patches).reshape(
            (new_crops.shape[0], new_crops.shape[1], 3, self._native_patch_size[0], self._native_patch_size[1]))
        coords = torch.stack(coords).reshape((new_crops.shape[0], new_crops.shape[1], 4))
        return patches, coords

    @torch.no_grad()
    def sample(self, images_cpu: Tensor, shared_memory: SharedMemory):
        new_crops: Tensor = shared_memory.action.clone().cpu()
        new_crops = self._calculate_crops(new_crops, images_cpu.shape)
        patches, coords = self._crop_and_resize(new_crops, images_cpu)
        shared_memory.add_glimpse(patches, coords)
