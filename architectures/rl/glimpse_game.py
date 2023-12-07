import abc
from typing import Optional, Tuple, Iterator

import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize


class InteractiveInPlaceSampler:
    def __init__(self, images: torch.Tensor, glimpse_grid_size, max_glimpses, native_patch_size=(16, 16),
                 max_size_ratio=0.5):
        # B x C x H x W
        assert len(images.shape) == 4 and images.shape[1] == 3
        self._images = images
        self._images_cpu = images.cpu()
        self._batch_size = images.shape[0]
        self._native_patch_size = native_patch_size
        self._glimpse_grid_size = glimpse_grid_size
        self._patches_per_glimpse = glimpse_grid_size ** 2
        self._max_glimpses = max_glimpses
        self._max_patches = max_glimpses * self._patches_per_glimpse
        self._max_size_ratio = max_size_ratio

        self._patches = torch.zeros(
            size=(self._batch_size, self._max_patches, 3, self._native_patch_size[0], self._native_patch_size[1]),
            dtype=self._images.dtype,
            device=self._images.device,
        )
        self._coords = torch.zeros(
            size=(self._batch_size, self._max_patches, 4),
            dtype=torch.long,
            device=self._images.device,
        )
        self._num_patches = 0

    def sample(self, action):
        # B x (y, x, s)
        new_crops = action.clone().cpu()
        max_crop_size = (
                min(self._images.shape[-2], self._images.shape[-1]) * self._max_size_ratio)
        min_crop_size = min(self._native_patch_size) * self._glimpse_grid_size

        new_crops[..., 2] = new_crops[..., 2] * (max_crop_size - min_crop_size) + min_crop_size
        new_crops[..., 0] = new_crops[..., 0] * (self._images.shape[-2] - new_crops[..., 2])
        new_crops[..., 1] = new_crops[..., 1] * (self._images.shape[-1] - new_crops[..., 2])
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
        return self._crop_and_resize(new_crops)

    def _crop_and_resize(self, new_crops: torch.Tensor):
        # rework for batched operation in future
        for b_idx in range(self._batch_size):
            for p_idx in range(new_crops.shape[1]):
                y, x, s = new_crops[b_idx, p_idx]
                patch = self._images_cpu[b_idx, :, y:y + s, x:x + s]

                if patch.shape[-1] != self._native_patch_size[0] or patch.shape[-2] != self._native_patch_size[1]:
                    patch = resize(
                        patch, list(self._native_patch_size),
                        antialias=False,
                        interpolation=InterpolationMode.BILINEAR
                    )
                self._patches[b_idx, self._num_patches + p_idx].copy_(patch, non_blocking=True)
                coords = torch.tensor([y, x, y + s, x + s], dtype=torch.long)
                self._coords[b_idx, self._num_patches + p_idx].copy_(coords, non_blocking=True)

        self._num_patches += new_crops.shape[1]

    @property
    def patches(self):
        return self._patches[:, :self._num_patches]

    @property
    def coords(self):
        return self._coords[:, :self._num_patches]

    @property
    def images(self):
        return self._images

    @property
    def num_patches(self):
        return self._num_patches


class GlimpseGameModelWrapper(abc.ABC):
    @abc.abstractmethod
    def __call__(self, images: torch.Tensor, patches: torch.Tensor,
                 coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented


class GlimpseGameEnv(EnvBase):

    def __init__(self, glimpse_model: GlimpseGameModelWrapper, dataloader: torch.utils.data.DataLoader,
                 state_dim: int, num_glimpses: int, batch_size: int, device):
        super().__init__(device=device)

        self.num_glimpses = num_glimpses
        self.glimpse_model = glimpse_model
        self.patch_sampler_class = InteractiveInPlaceSampler

        self.state_dim = state_dim
        self.batch_size = torch.Size((batch_size,))

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=torch.Size((batch_size, self.state_dim))),
            step=DiscreteTensorSpec(n=self.num_glimpses, shape=torch.Size((batch_size, 1))),
            mae_loss=UnboundedContinuousTensorSpec(shape=torch.Size((batch_size, 1))),
            shape=self.batch_size
        )
        self.action_spec = BoundedTensorSpec(low=0., high=1., shape=torch.Size((batch_size, 3)))
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size((batch_size, 1)))

        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)

        self.sampler = None
        self._last_final_sampler = None
        self.current_state = None
        self.current_error = None
        self.n_step = 0
        self.last_final_rmse = 0.

    def __len__(self):
        return len(self.dataloader) * self.num_glimpses

    class WithPolicy:
        def __init__(self, env, policy):
            self.env = env
            self.policy = policy

        def __len__(self):
            return len(self.env)

        def __iter__(self):
            return self

        def __next__(self):
            return self.env.rollout(max_steps=self.env.num_glimpses + 1, policy=self.policy)

    def iterate_with_policy(self, policy):
        return self.WithPolicy(self, policy)

    def pop_prev_sampler(self):
        sampler = self._last_final_sampler
        self._last_final_sampler = None
        return sampler

    def reset_dataloader(self):
        self.dataloader_iter = iter(self.dataloader)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        assert self.dataloader
        self.n_step = 0

        batch = next(self.dataloader_iter)  # will throw StopIteration, let it propagate

        self.images = batch['image'].to(self.device)
        self.sampler = self.patch_sampler_class(self.images, glimpse_grid_size=2, max_glimpses=self.num_glimpses)
        self.current_state, self.current_error = self.glimpse_model(images=self.images, patches=self.sampler.patches,
                                                                    coords=self.sampler.coords)

        return TensorDict({
            'observation': self.current_state,
            'step': torch.ones_like(self.current_error, dtype=torch.long) * self.n_step,
            'mae_loss': self.current_error,
        }, batch_size=self.images.shape[0], device=self.device)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        assert self.sampler

        action = tensordict['action']
        self.sampler.sample(action=action)

        self.current_state, loss = self.glimpse_model(images=self.images, patches=self.sampler.patches,
                                                      coords=self.sampler.coords)

        reward = self.current_error - loss
        self.current_error = loss

        self.n_step += 1

        if self.is_done:
            self.last_final_rmse = loss.mean().item()
            self._last_final_sampler = self.sampler

        return TensorDict({
            'observation': self.current_state,
            'step': torch.ones_like(self.current_error, dtype=torch.long) * self.n_step,
            'reward': reward,
            'mae_loss': self.current_error,
            'done': (torch.ones_like if self.is_done else torch.zeros_like)(self.current_error, dtype=torch.bool)
        }, batch_size=self.images.shape[0], device=self.device)

    @property
    def is_done(self):
        return self.n_step >= self.num_glimpses

    def _set_seed(self, seed: Optional[int]):
        pass


class GlimpseGameDataCollector(SyncDataCollector):
    def __init__(self, env, *args, **kwargs):
        assert isinstance(env, GlimpseGameEnv)
        super().__init__(env, *args, **kwargs, total_frames=-1)

    def __len__(self):
        return len(self.env)

    def __iter__(self) -> Iterator[TensorDictBase]:
        self.env.reset_dataloader()
        return super().__iter__()
