import abc
from typing import Optional, Tuple, Iterator

import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase

from datasets.patch_sampler import InteractiveSampler


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
        self.patch_sampler_class = InteractiveSampler

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
        self.sampler = self.patch_sampler_class(self.images, init_grid=False)
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
        self.sampler.sample_multi_relative(new_crops=action, grid_size=2)

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
