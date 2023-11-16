import sys
from typing import Optional

import torch
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, \
    ReplayBuffer, LazyTensorStorage
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import SACLoss, SoftUpdate
from tqdm import tqdm

from architectures.glimpse_selectors import ElasticAttentionMapEntropy
from architectures.mae import MaskedAutoencoderViT, mae_vit_base_patch16
from architectures.utils import rev_normalize
from datasets import ImageNet1k
from datasets.patch_sampler import InteractiveSampler


class ActorCritic(nn.Module):
    def __init__(self, input_dim=196, hidden_dim=256, action_dim=3, num_glimpses=12):
        super().__init__()

        class ActorNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.observation_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.step_net = nn.Sequential(
                    nn.Linear(num_glimpses, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.actor_net = nn.Sequential(
                    nn.Linear(3 * hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2 * action_dim),
                    NormalParamExtractor(scale_mapping="biased_softplus_0.5"),
                )

            def forward(self, observation, step):
                observation = self.observation_net(observation)
                step = torch.nn.functional.one_hot(step.clip(0, num_glimpses - 1), num_classes=num_glimpses).to(
                    observation.dtype)
                step = self.step_net(step.squeeze(1))
                return self.actor_net(torch.cat([observation, step], dim=-1))

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                ActorNet(),
                in_keys=["observation", "step"],
                out_keys=["loc", "scale"]
            ),
            spec=BoundedTensorSpec(
                minimum=0,
                maximum=1,
                shape=action_dim
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": 0,
                "max": 1,
            },
            return_log_prob=False,
            default_interaction_type=InteractionType.RANDOM
        )

        class QValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.observation_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.step_net = nn.Sequential(
                    nn.Linear(num_glimpses, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.action_net = nn.Sequential(
                    nn.Linear(action_dim, hidden_dim // 2),
                    nn.ReLU()
                )
                self.value_net = nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, observation, step, action):
                observation = self.observation_net(observation)
                step = torch.nn.functional.one_hot(step.clip(0, num_glimpses - 1), num_classes=num_glimpses).to(
                    observation.dtype)
                step = self.step_net(step.squeeze(1))
                action = self.action_net(action)
                return self.value_net(torch.cat([observation, step, action], dim=-1))

        self.qvalue_module = ValueOperator(
            module=QValueNet(),
            in_keys=["observation", "step", "action"],
        )


class GlimpseGameEnv(EnvBase):

    def __init__(self, num_glimpses, image_size, pretrained_mae_path, batch_size: int = 16):
        super().__init__()
        self.num_glimpses = num_glimpses

        self.mae: MaskedAutoencoderViT = mae_vit_base_patch16(img_size=image_size, out_chans=3)
        self.mae = torch.compile(self.mae, mode='reduce-overhead')

        self.extractor = ElasticAttentionMapEntropy(self)
        self.patch_sampler_class = InteractiveSampler

        self.state_dim = 14 * 14  # entropy map
        self.action_dim = 3  # x y s
        self.batch_size = torch.Size((batch_size,))

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=torch.Size((batch_size, self.state_dim))),
            step=DiscreteTensorSpec(n=self.num_glimpses, shape=torch.Size((batch_size, 1))),
            mae_loss=UnboundedContinuousTensorSpec(shape=torch.Size((batch_size, 1))),
            shape=self.batch_size
        )
        self.action_spec = BoundedTensorSpec(low=0., high=1., shape=torch.Size((batch_size, self.action_dim)))
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size((batch_size, 1)))

        print(self._load_pretrained_elastic(pretrained_mae_path), file=sys.stderr)

        # freeze mae weights
        for param in self.mae.parameters():
            param.requires_grad = False
        self.mae.eval()

        self.dataloader = None
        self.dataloader_iter = None
        self.auto_reset_dataloader = False

        self.sampler = None
        self.current_state = None
        self.current_error = None
        self.n_step = 0
        self.last_final_rmse = 0

    def _load_pretrained_elastic(self, path=""):
        checkpoint = torch.load(path, map_location='cpu')["state_dict"]
        return self.load_state_dict(checkpoint, strict=False)

    def set_dataloader(self, dataloader, auto_reset_dataloader=False):
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.auto_reset_dataloader = auto_reset_dataloader

    def __len__(self):
        return len(self.dataloader)

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

    def _calculate_loss(self, out):
        pred = self.mae.unpatchify(out)
        pred = rev_normalize(pred)
        target = rev_normalize(self.images)
        return torch.sqrt(
            torch.nn.functional.mse_loss(pred, target, reduce=False)
            .reshape(pred.shape[0], -1)
            .mean(dim=-1, keepdim=True)
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        assert self.dataloader

        self.sampler = None
        self.current_state = None
        self.current_error = None
        self.n_step = 0

        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            if self.auto_reset_dataloader:
                self.dataloader_iter = iter(self.dataloader)
                batch = next(self.dataloader_iter)
            else:
                raise

        self.images = batch['image'].to(self.device)
        self.sampler = self.patch_sampler_class(self.images, init_grid=False)

        with torch.no_grad():
            latent = self.mae.forward_encoder(self.sampler.patches, coords=self.sampler.coords)
            out = self.mae.forward_decoder(latent)
            self.current_error = self._calculate_loss(out)
            self.current_state = self.extractor().reshape(self.images.shape[0], -1)

        return TensorDict({
            'observation': self.current_state,
            'step': torch.ones_like(self.current_error, dtype=torch.long) * self.n_step,
            'mae_loss': self.current_error,
        }, batch_size=self.images.shape[0], device=self.device)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict['action']
        self.sampler.sample_multi_relative(new_crops=action, grid_size=2)

        with torch.no_grad():
            latent = self.mae.forward_encoder(self.sampler.patches, coords=self.sampler.coords)
            out = self.mae.forward_decoder(latent)
            loss = self._calculate_loss(out)
            self.current_state = self.extractor().reshape(self.images.shape[0], -1)

        reward = self.current_error - loss
        self.current_error = loss
        # reward = reward.clip(min=0, max=None)

        self.n_step += 1
        is_done = self.n_step >= self.num_glimpses
        if is_done:
            self.last_final_rmse = loss.mean().item()

        return TensorDict({
            'observation': self.current_state,
            'step': torch.ones_like(self.current_error, dtype=torch.long) * self.n_step,
            'reward': reward,
            'mae_loss': self.current_error,
            'done': (torch.ones_like if is_done else torch.zeros_like)(self.current_error, dtype=torch.bool)
        }, batch_size=self.images.shape[0], device=self.device)

    def _set_seed(self, seed: Optional[int]):
        pass


def train(base_env, env, model, loss_module, train_loader, target_net_updater):
    base_env.set_dataloader(train_loader)

    model.train()

    total_frames = 1000000
    num_iters = 8
    frames_per_batch = 32
    buffer_size = 100000
    iter_batch_size = 256
    init_random_frames = iter_batch_size * 100

    lr = 1.0e-3
    weight_decay = 0.0
    adam_eps = 1.0e-8

    collector = SyncDataCollector(
        env,
        model.policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device='cuda',
        init_random_frames=init_random_frames
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=buffer_size
        ),
        prefetch=3,
        batch_size=iter_batch_size
    )

    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=lr,
        weight_decay=weight_decay,
        eps=adam_eps,
    )
    scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_actor, total_frames // frames_per_batch, 1e-8
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=lr,
        weight_decay=weight_decay,
        eps=adam_eps,
    )
    scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_critic, total_frames // frames_per_batch, 1e-8
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=lr,
        weight_decay=weight_decay,
        eps=adam_eps,
    )
    scheduler_alpha = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_alpha, total_frames // frames_per_batch, 1e-8
    )

    pbar = tqdm(total=total_frames)

    for i, tensordict_data in enumerate(collector):
        collector.update_policy_weights_()

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())

        pbar.update(tensordict_data.numel())

        if len(replay_buffer) > init_random_frames:
            for _ in range(num_iters):
                subdata = replay_buffer.sample()
                loss_td = loss_module(subdata.to('cuda'))

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                # torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                # Update actor
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                # Update critic
                optimizer_critic.zero_grad()
                q_loss.backward()
                optimizer_critic.step()

                # Update alpha
                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()

                target_net_updater.step()

                pbar.set_description(", ".join(
                    [f'actor loss {actor_loss.mean().item()}', f'qvalue loss {q_loss.mean().item()}',
                     f'alpha loss {alpha_loss.mean().item()}', f'f_rmse {base_env.last_final_rmse}',
                     f'lr {optimizer_actor.param_groups[0]["lr"]}']))
        scheduler_actor.step()
        scheduler_critic.step()
        scheduler_alpha.step()
    torch.save(model.state_dict(), 'ppo5.pth')


def validate(base_env, env, model, val_loader):
    model.eval()

    base_env.set_dataloader(val_loader, auto_reset_dataloader=False)

    score = []

    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        for eval_rollout in tqdm(env.iterate_with_policy(model.policy_module)):
            mean_mae_loss = eval_rollout['mae_loss'][:, -1].mean().item()
            score.append(mean_mae_loss)

    print(sum(score) / len(score))


if __name__ == '__main__':
    batch_size = 32

    base_env = GlimpseGameEnv(num_glimpses=12, image_size=(224, 224), pretrained_mae_path='../elastic_mae.ckpt',
                              batch_size=batch_size).to('cuda')

    env = base_env
    data = ImageNet1k(data_dir='/home/adam/datasets/imagenet', train_batch_size=batch_size, eval_batch_size=batch_size,
                      num_workers=8, always_drop_last=True)
    data.setup('fit')
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    model = ActorCritic().to('cuda')

    loss_module = SACLoss(actor_network=model.policy_module, qvalue_network=model.qvalue_module,
                          loss_function='smooth_l1', delay_actor=False,
                          delay_qvalue=True, alpha_init=1.0)
    loss_module.make_value_estimator(gamma=0.99)
    target_net_updater = SoftUpdate(loss_module, eps=0.995)

    train(base_env, env, model, loss_module, train_loader, target_net_updater)
    validate(base_env, env, model, val_loader)
