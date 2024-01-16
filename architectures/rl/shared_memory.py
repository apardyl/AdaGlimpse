from ctypes import Structure, c_int
from multiprocessing import Value
from typing import Tuple, Dict, Callable

import torch
from tensordict import TensorDict
from torch import Tensor


class _SharedMemStruct(Structure):
    _fields_ = [('current_batch_size', c_int), ('current_glimpse', c_int)]


class SharedMemory:
    def __init__(self, max_glimpses: int, image_size: Tuple[int, int], native_patch_size: Tuple[int, int],
                 glimpse_grid_size: int, batch_size: int, device: torch.device,
                 create_target_tensor_fn: Callable[[int], Tensor],
                 copy_target_tensor_fn: Callable[[Tensor, Dict[str, Tensor]], None]):
        self._max_glimpses = max_glimpses
        self._glimpse_grid_size = glimpse_grid_size
        self._patches_per_glimpse = glimpse_grid_size ** 2
        self._max_patches = max_glimpses * self._patches_per_glimpse
        self._copy_target_tensor_fn = copy_target_tensor_fn

        self._shared = TensorDict({
            "images": torch.zeros((batch_size, 3, image_size[0], image_size[1])),
            "patches": torch.zeros((batch_size, self._max_patches, 3, native_patch_size[0], native_patch_size[1])),
            "coords": torch.zeros((batch_size, self._max_patches, 4)),
            "mask": torch.ones((batch_size, self._max_patches, 1)),
            "action": torch.zeros((batch_size, 3)),
            "target": create_target_tensor_fn(batch_size),
            "done": torch.zeros((batch_size, 1), dtype=torch.bool)
        }, batch_size=(), device=device)

        self._state = Value(_SharedMemStruct, lock=True)
        self._shared.share_memory_()

    def close(self):
        del self._shared
        del self._state

    def _get_state(self) -> Tuple[int, int]:
        s = self._state.get_obj()
        return s.current_batch_size, s.current_glimpse

    @property
    def current_patches(self):
        current_batch_size, current_glimpse = self._get_state()
        return self._shared["patches"][:current_batch_size, :current_glimpse * self._patches_per_glimpse]

    @property
    def all_patches(self):
        return self._shared["patches"][:self.current_batch_size]

    @property
    def current_coords(self):
        current_batch_size, current_glimpse = self._get_state()
        return self._shared["coords"][:current_batch_size, :current_glimpse * self._patches_per_glimpse]

    @property
    def all_coords(self):
        return self._shared["coords"][:self.current_batch_size]

    @property
    def images(self):
        return self._shared['images'][:self.current_batch_size]

    @property
    def current_mask(self):
        current_batch_size, current_glimpse = self._get_state()
        return self._shared["mask"][:current_batch_size, :current_glimpse * self._patches_per_glimpse]

    @property
    def all_mask(self):
        return self._shared['mask'][:self.current_batch_size]

    @property
    def action(self):
        return self._shared['action'][:self.current_batch_size]

    @action.setter
    def action(self, value: Tensor):
        assert value.max() <= 1. and value.min() >= 0, 'action must be in range 0-1'
        self._shared["action"][:self.current_batch_size].copy_(value)

    @property
    def target(self):
        return self._shared['target'][:self.current_batch_size]

    @property
    def done(self):
        return self._shared['done'][:self.current_batch_size]

    @done.setter
    def done(self, value: Tensor):
        self._shared["done"][:self.current_batch_size].copy_(value)

    @property
    def current_glimpse(self):
        return self._state.current_glimpse

    @current_glimpse.setter
    def current_glimpse(self, value: int):
        self._state.current_glimpse = value

    @property
    def current_batch_size(self):
        return self._state.current_batch_size

    @current_batch_size.setter
    def current_batch_size(self, value: int):
        self._state.current_batch_size = value

    @property
    def is_done(self):
        return self.current_glimpse >= self._max_glimpses or torch.all(self.done)

    def set_batch(self, batch: Dict[str, Tensor]):
        b = batch['image'].shape[0]
        self.current_batch_size = b
        self._shared["images"][:b].copy_(batch['image'])
        self.current_glimpse = 0
        self._copy_target_tensor_fn(self._shared["target"], batch)
        self._shared['coords'][:b].fill_(0)
        self._shared["mask"][:b].fill_(1)
        self._shared["done"][:b].fill_(False)

    def add_glimpse(self, new_patches: Tensor, new_coords: Tensor, new_masks: Tensor):
        b, g = self.current_batch_size, self.current_glimpse
        assert new_patches.shape[0] == b
        assert new_patches.shape[1] == self._patches_per_glimpse

        start = g * self._patches_per_glimpse
        end = start + self._patches_per_glimpse

        self._shared["patches"][:b, start: end].copy_(new_patches)
        self._shared["coords"][:b, start: end].copy_(new_coords)
        self._shared["mask"][:b, start: end].copy_(new_masks)

        self.current_glimpse = g + 1

        if self.current_glimpse >= self._max_glimpses:
            self._shared["done"][:b].fill_(True)
