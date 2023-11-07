import sys
from functools import partial
from typing import Any

import torch
import torchmetrics
from tensordict import TensorDict
from torch import nn
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from architectures.base import BaseArchitecture
from architectures.glimpse_selectors import ElasticAttentionMapEntropy
from architectures.mae import mae_vit_base_patch16, MaskedAutoencoderViT
from architectures.rl_modules import Actor, Critic, OrnsteinUhlenbeckActionNoise
from architectures.utils import MetricMixin, MaeScheduler
from datasets.base import BaseDataModule
from datasets.patch_sampler import InteractiveSampler
from datasets.utils import IMAGENET_MEAN, IMAGENET_STD


class SmartGlimpse(BaseArchitecture, MetricMixin):
    def __init__(self, datamodule: BaseDataModule, out_chans=3, pretrained_mae_path=None, num_glimpses=12,
                 rl_iters_per_step=1, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.num_glimpses = num_glimpses
        self.rl_iters_per_step = rl_iters_per_step

        self.automatic_optimization = False  # disable lightning automation

        self.mae: MaskedAutoencoderViT = mae_vit_base_patch16(img_size=datamodule.image_size, out_chans=out_chans)

        self.extractor = ElasticAttentionMapEntropy(self)
        self.patch_sampler_class = InteractiveSampler

        self.state_dim = 14 * 14  # entropy map
        self.action_dim = 3  # x y s

        self.actor = Actor(self.state_dim, self.action_dim)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_loss = nn.MSELoss()

        self.buffer = TensorDictReplayBuffer(storage=LazyMemmapStorage(max_size=10000, device=self.device),
                                             batch_size=1024)

        if self.compile_model:
            self.mae = torch.compile(self.mae, mode='reduce-overhead')

        if pretrained_mae_path:
            print(self.load_pretrained_elastic(pretrained_mae_path), file=sys.stderr)

            # freeze mae weights for now
            for param in self.mae.parameters():
                param.requires_grad = False
            self.mae.eval()

        self.define_metric('rmse', partial(torchmetrics.MeanSquaredError, squared=False))
        self.register_buffer('imagenet_mean', torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1))
        # self.define_metric('accuracy',
        #                    partial(torchmetrics.classification.MulticlassAccuracy,
        #                            num_classes=self.num_classes,
        #                            average='micro'))
        # self.criterion = nn.CrossEntropyLoss()

        self.noise = OrnsteinUhlenbeckActionNoise(action_dim=self.action_dim)

    def configure_optimizers(self):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.min_lr)
        actor_scheduler = MaeScheduler(
            optimizer=actor_optimizer,
            lr=self.lr,
            warmup_epochs=self.warmup_epochs,
            min_lr=self.min_lr,
            epochs=self.epochs
        )
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.min_lr)
        critic_scheduler = MaeScheduler(
            optimizer=critic_optimizer,
            lr=self.lr,
            warmup_epochs=self.warmup_epochs,
            min_lr=self.min_lr,
            epochs=self.epochs
        )

        return (
            {
                "optimizer": actor_optimizer,
                "lr_scheduler": {
                    'scheduler': actor_scheduler,
                    'interval': 'epoch'
                }
            },
            {
                "optimizer": critic_optimizer,
                "lr_scheduler": {
                    'scheduler': critic_scheduler,
                    'interval': 'epoch'
                }
            }
        )

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(SmartGlimpse.__name__)
        parser.add_argument('--num-glimpses',
                            help='number of glimpses to take',
                            type=int,
                            default=12)
        parser.add_argument('--pretrained-mae-path',
                            help='path to pretrained MAE weights',
                            type=str,
                            default='elastic_mae.ckpt')
        parser.add_argument('--rl-iters-per-step',
                            help='RL loop iterations per training step',
                            type=int,
                            default=1)
        return parent_parser

    def load_pretrained_elastic(self, path=""):
        checkpoint = torch.load(path, map_location='cpu')["state_dict"]
        return self.load_state_dict(checkpoint, strict=False)

    def __rev_normalize(self, img):
        return torch.clip((img * self.imagenet_std + self.imagenet_mean) * 255, 0, 255)

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)

        pred = self.mae.unpatchify(out['out'])
        pred = self.__rev_normalize(pred)
        target = self.__rev_normalize(batch['image'])

        self.log_metric(mode, 'rmse', pred, target)

    def forward(self, batch) -> Any:
        images = batch['image']
        labels = batch['label']

        critic_optimizer, actor_optimizer = self.optimizers()

        env = self.patch_sampler_class(images=images)

        state = torch.zeros((images.shape[0], self.state_dim), device=images.device)
        out = None
        loss = None

        for glimpse_idx in range(self.num_glimpses):
            with torch.no_grad():
                action = self.actor(state)
                if self.training:
                    noise = torch.from_numpy(self.noise.sample()).to(action.dtype).to(action.device)
                    action = torch.clamp(action + noise, 0, 1)
                env.sample_multi_relative(new_crops=action, grid_size=2)
                latent = self.mae.forward_encoder(env.patches, coords=env.coords)
                out = self.mae.forward_decoder(latent)
                # pred = self.mae.forward_head(latent)
                rec_loss = self.mae.forward_reconstruction_loss(images, out, mean=False)
                # cls_loss = self.criterion(pred, labels)
                loss = rec_loss
                next_state = self.extractor().reshape(images.shape[0], -1)
            if self.training:
                # detach all gradients between MAE and RL
                self._on_train_glimpse(glimpse_idx, critic_optimizer, actor_optimizer, state.detach().clone(),
                                       action.detach().clone(),
                                       loss.detach().clone(), next_state.detach().clone())
            state = next_state

        return {'out': out, 'loss': loss.mean(), 'coords': env.coords}

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.lr_schedulers() is not None:
            for sch in self.lr_schedulers():
                sch.step(self.current_epoch)
        self.buffer.empty()

    def _on_train_glimpse(self, glimpse_idx, critic_optimizer, actor_optimizer, current_state, current_action,
                          current_loss, current_next_state):
        self.buffer.extend(TensorDict({
            'state': current_state,
            'action': current_action,
            'reward': (-current_loss / 100),  # normalize rewards
            'next_state': current_next_state
        }, batch_size=current_state.shape[0]))

        critic_losses = []
        actor_losses = []

        for _ in range(self.rl_iters_per_step):
            sample = self.buffer.sample()
            state, action, reward, next_state = (
                sample['state'].to(current_state.device),
                sample['action'].to(current_state.device),
                sample['reward'].to(current_state.device),
                sample['next_state'].to(current_state.device)
            )

            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()

            # optimize critic
            next_action = self.actor(next_state).detach()
            next_val = self.critic(next_state, next_action).detach()
            expected_val = reward + 0.99 * next_val
            predicted_val = self.critic(state, action)
            loss_critic = self.critic_loss(predicted_val, expected_val)
            assert torch.isfinite(loss_critic)
            # loss_critic = torch.clip(loss_critic, -10, 10)  # prevent exploding gradient
            loss_critic.backward()
            critic_optimizer.step()
            critic_losses.append(loss_critic.item())

            # optimize actor
            predicted_action = self.actor(state)
            loss_actor = -1 * torch.sum(self.critic(state, predicted_action))
            assert torch.isfinite(loss_critic)
            # loss_actor = torch.clip(loss_actor, -10, 10)  # prevent exploding gradient
            loss_actor.backward()
            actor_optimizer.step()
            actor_losses.append(loss_actor.item())

        self.log('train/critic_loss', sum(critic_losses) / len(critic_losses), on_step=True, on_epoch=False,
                 sync_dist=False)
        self.log('train/actor_loss', sum(actor_losses) / len(actor_losses), on_step=True, on_epoch=False,
                 sync_dist=False)
