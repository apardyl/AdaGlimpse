from typing import Tuple, Dict

import torch
from tensordict import TensorDict
from torch import Tensor


class SharedMemory:
    def __init__(self, max_glimpses: int, image_size: Tuple[int, int], native_patch_size: Tuple[int, int],
                 glimpse_grid_size: int, batch_size: int, device: torch.device):
        self._max_glimpses = max_glimpses
        self._glimpse_grid_size = glimpse_grid_size
        self._patches_per_glimpse = glimpse_grid_size ** 2
        self._max_patches = max_glimpses * self._patches_per_glimpse

        self._shared = TensorDict({
            "images": torch.zeros((batch_size, 3, image_size[0], image_size[1])),
            "patches": torch.zeros((batch_size, self._max_patches, 3, native_patch_size[0], native_patch_size[1])),
            "coords": torch.zeros((batch_size, self._max_patches, 4)),
            "mask": torch.ones((batch_size, self._max_patches, 1)),
            "action": torch.zeros((batch_size, 3)),
            "current_batch_size": torch.zeros(1, dtype=torch.long),
            "current_glimpse": torch.zeros(1, dtype=torch.long)
        }, batch_size=(), device=device)

        self._shared.share_memory_()

    @property
    def patches(self):
        return self._shared["patches"][:self.current_batch_size, :self.current_glimpse * self._patches_per_glimpse]

    @property
    def coords(self):
        return self._shared["coords"][:self.current_batch_size, :self.current_glimpse * self._patches_per_glimpse]

    @property
    def images(self):
        return self._shared['images'][:self.current_batch_size]

    @property
    def mask(self):
        return self._shared['mask'][:self.current_batch_size]

    @property
    def action(self):
        return self._shared['action'][:self.current_batch_size]

    @property
    def current_glimpse(self):
        return int(self._shared['current_glimpse'])

    @current_glimpse.setter
    def current_glimpse(self, value: int):
        self._shared["current_glimpse"][:].fill_(value)

    @property
    def current_batch_size(self):
        return int(self._shared['current_batch_size'])

    @current_batch_size.setter
    def current_batch_size(self, value: int):
        self._shared['current_batch_size'][:].fill_(value)

    @property
    def is_done(self):
        return self.current_glimpse >= self._max_glimpses

    def set_batch(self, batch: Dict[str, Tensor]):
        b = batch['image'].shape[0]
        self.current_batch_size = b
        self._shared["images"][:b].copy_(batch['image'])
        self.current_glimpse = 0

    def add_glimpse(self, new_patches: Tensor, new_coords: Tensor):
        b, g = self.current_batch_size, self.current_glimpse
        assert new_patches.shape[0] == b
        assert new_patches.shape[1] == self._patches_per_glimpse

        start = g * self._patches_per_glimpse
        end = start + self._patches_per_glimpse

        try:

            self._shared["patches"][:b, start: end].copy_(new_patches)
            self._shared["coords"][:b, start: end].copy_(new_coords)
            self._shared["mask"][:b, start: end].fill_(0)

        except Exception as e:
            print('xd')
            raise e

        self.current_glimpse = g + 1

    def set_action(self, action: Tensor):
        # noinspection PyTypeChecker
        self._shared["action"].copy_(action)
