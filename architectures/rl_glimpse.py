import sys
from contextlib import nullcontext
from typing import Optional

import torch
from lightning.pytorch.strategies import ParallelStrategy
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tensordict import TensorDict
from torch.utils.data import DistributedSampler
from torchrl.data import ReplayBuffer, LazyTensorStorage, BoundedTensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SACLoss, SoftUpdate

from architectures.base import AutoconfigLightningModule
from architectures.mae import MaskedAutoencoderViT, mae_vit_base_patch16
from architectures.rl.glimpse_engine import SyncGlimpseEngine
from architectures.rl.shared_memory import SharedMemory
from architectures.rl.transformer_actor_critic import TransformerActorCritic
from architectures.utils import MetricMixin, RevNormalizer
from datasets.base import BaseDataModule


class RlMAE(AutoconfigLightningModule, MetricMixin):
    internal_data = True
    checkpoint_metric = 'val/mae_rmse'

    rl_state_dim = 768
    rl_hidden_dim = 256
    rl_action_dim = 3

    def __init__(self, datamodule: BaseDataModule, pretrained_mae_path=None, num_glimpses=14,
                 rl_iters_per_step=1, epochs=100, init_random_batches=100, init_backbone_batches=50000,
                 rl_batch_size=64, replay_buffer_size=10000, lr=3e-4, backbone_lr=1e-5, **_) -> None:
        super().__init__()

        self.steps_per_epoch = None
        self.num_glimpses = num_glimpses
        self.rl_iters_per_step = rl_iters_per_step

        self.batch_size = max(datamodule.train_batch_size, datamodule.eval_batch_size)

        self.rl_batch_size = rl_batch_size
        self.epochs = epochs
        self.init_random_batches = init_random_batches
        self.init_backbone_batches = init_backbone_batches
        self.patches_per_glimpse = 4
        self.lr = lr
        self.backbone_lr = backbone_lr

        self.replay_buffer_size = replay_buffer_size

        self.datamodule = datamodule

        self.automatic_optimization = False  # disable lightning automation

        self.mae = mae_vit_base_patch16(img_size=datamodule.image_size, out_chans=3)
        # noinspection PyTypeChecker
        self.mae: MaskedAutoencoderViT = torch.compile(self.mae, mode='reduce-overhead')

        if pretrained_mae_path:
            self.load_pretrained_elastic(pretrained_mae_path)

        self.actor_critic = TransformerActorCritic(action_dim=3, embed_dim=self.rl_state_dim)

        self.rl_loss_module = SACLoss(actor_network=self.actor_critic.policy_module,
                                      qvalue_network=self.actor_critic.qvalue_module,
                                      loss_function='smooth_l1',
                                      delay_actor=False,
                                      delay_qvalue=True,
                                      alpha_init=1.0)
        self.rl_loss_module.make_value_estimator(gamma=0.99)
        self.target_net_updater = SoftUpdate(self.rl_loss_module, eps=0.995)

        self.train_loader = None
        self.val_loader = None
        self.replay_buffer = None
        self.rev_normalizer = RevNormalizer()
        self.game_state = [None]
        self.add_pos_embed = True

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(RlMAE.__name__)
        parser.add_argument('--lr',
                            help='learning-rate',
                            type=float,
                            default=3e-4)
        parser.add_argument('--backbone-lr',
                            help='backbone learning-rate',
                            type=float,
                            default=1e-5)
        parser.add_argument('--pretrained-mae-path',
                            help='path to pretrained MAE weights',
                            type=str,
                            default='elastic_mae.ckpt')
        parser.add_argument('--epochs',
                            help='number of epochs',
                            type=int,
                            default=100)
        parser.add_argument('--num-glimpses',
                            help='number of glimpses to take',
                            type=int,
                            default=14)
        parser.add_argument('--rl-iters-per-step',
                            help='number of rl iterations per step',
                            type=int,
                            default=1)
        parser.add_argument('--init-random-batches',
                            help='number of random action batches on training start',
                            type=int,
                            default=100)
        parser.add_argument('--init-backbone-batches',
                            help='number of rl pre-training steps before starting to train the backbone',
                            type=int,
                            default=50000)
        parser.add_argument('--rl-batch-size',
                            help='batch size of the rl loop',
                            type=int,
                            default=64)
        parser.add_argument('--replay-buffer-size',
                            help='rl replay buffer size in episodes',
                            type=int,
                            default=10000)
        return parent_parser

    def configure_optimizers(self):
        critic_params = list(self.rl_loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(self.rl_loss_module.actor_network_params.flatten_keys().values())

        actor_optimizer = torch.optim.Adam(actor_params, self.lr)
        actor_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=actor_optimizer,
            max_lr=self.lr,
            pct_start=0.01,
            div_factor=1,
            final_div_factor=1000,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        critic_optimizer = torch.optim.Adam(critic_params, self.lr)
        critic_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=critic_optimizer,
            max_lr=self.lr,
            pct_start=0.01,
            div_factor=1,
            final_div_factor=1000,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        alpha_optimizer = torch.optim.Adam([self.rl_loss_module.log_alpha], self.lr)
        alpha_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=alpha_optimizer,
            max_lr=self.lr,
            pct_start=0.01,
            div_factor=1,
            final_div_factor=1000,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        mae_optimizer = torch.optim.Adam(self.mae.parameters(), self.lr)
        mae_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=mae_optimizer,
            max_lr=self.backbone_lr,
            pct_start=0.01,
            div_factor=1,
            final_div_factor=1000,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )

        return (
            {
                "optimizer": actor_optimizer,
                "lr_scheduler": {
                    'scheduler': actor_scheduler,
                    'interval': 'step'
                }
            },
            {
                "optimizer": critic_optimizer,
                "lr_scheduler": {
                    'scheduler': critic_scheduler,
                    'interval': 'step'
                }
            },
            {
                "optimizer": alpha_optimizer,
                "lr_scheduler": {
                    'scheduler': alpha_scheduler,
                    'interval': 'step'
                }
            },
            {
                "optimizer": mae_optimizer,
                "lr_scheduler": {
                    'scheduler': mae_scheduler,
                    'interval': 'step'
                }
            }
        )

    def load_pretrained_elastic(self, path=""):
        checkpoint = torch.load(path, map_location='cpu')["state_dict"]
        checkpoint = {k[4:]: v for k, v in checkpoint.items() if k.startswith('mae.')}
        print(self.mae.load_state_dict(checkpoint, strict=False), file=sys.stderr)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return SyncGlimpseEngine(
            dataloader=self.train_loader,
            max_glimpses=self.num_glimpses,
            glimpse_grid_size=2,
            native_patch_size=(16, 16),
            batch_size=self.datamodule.train_batch_size,
            device=self.device,
            image_size=self.datamodule.image_size,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return SyncGlimpseEngine(
            dataloader=self.datamodule.val_dataloader(),
            max_glimpses=self.num_glimpses,
            glimpse_grid_size=2,
            native_patch_size=(16, 16),
            batch_size=self.datamodule.eval_batch_size,
            device=self.device,
            image_size=self.datamodule.image_size
        )

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: str) -> None:
        if stage != 'fit':
            raise NotImplemented()

        self.datamodule.setup(stage)
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loader = self.datamodule.val_dataloader()
        if isinstance(self.trainer.strategy, ParallelStrategy):
            self.train_loader.sampler = DistributedSampler(self.train_loader.dataset)
            self.val_loader.sampler = DistributedSampler(self.val_loader.dataset)
        self.steps_per_epoch = len(self.train_loader) * (self.num_glimpses + 1)
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.replay_buffer_size,
                device=self.device
            ),
            batch_size=self.rl_batch_size
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        # fix for torch rl removing caches on copy to device.
        if not hasattr(self.rl_loss_module, '_cache'):
            self.rl_loss_module._cache = {}

    def _calculate_score(self, out, images):
        pred = self.mae.unpatchify(out)
        pred = self.rev_normalizer(pred)
        target = self.rev_normalizer(images)
        return -torch.sqrt(
            torch.nn.functional.mse_loss(pred, target, reduce=False)
            .reshape(pred.shape[0], -1)
            .mean(dim=-1, keepdim=True)
        )

    def _calculate_loss(self, out, images):
        return self.mae.forward_reconstruction_loss(images, out)

    def forward_game_state(self, state: SharedMemory, with_loss_and_grad=False):
        with nullcontext() if with_loss_and_grad else torch.no_grad():
            images = state.images
            step = state.current_glimpse
            latent, pos_embed = self.mae.forward_encoder(state.patches, coords=state.coords)
            out = self.mae.forward_decoder(latent)
            loss = None
            if with_loss_and_grad:
                loss = self._calculate_loss(out, images)
            score = self._calculate_score(out, images)

            if self.add_pos_embed:
                latent[:, 1:] = latent[:, 1:] + pos_embed

            observation = torch.zeros(latent.shape[0], self.num_glimpses * self.patches_per_glimpse + 1,
                                      latent.shape[-1], device=latent.device, dtype=latent.dtype)
            observation[:, :latent.shape[1]] = latent
            observation.detach_()

        done = (torch.ones if step >= self.num_glimpses else torch.zeros)(size=(images.shape[0], 1),
                                                                          dtype=torch.bool, device=images.device)
        state = TensorDict({
            'observation': observation,
            'mask': state.mask,
            'step': torch.ones(size=(images.shape[0], 1), dtype=torch.long, device=images.device) * step,
            'done': done,
            'terminated': done,
            'score': score,
        }, batch_size=observation.shape[0])

        return state, step, loss

    @torch.no_grad()
    def forward_action(self, state_dict: TensorDict, exploration_type: ExplorationType):
        with set_exploration_type(exploration_type):
            action = self.actor_critic.policy_module(state_dict)['action']
            return action

    def random_action(self, batch_size):
        return BoundedTensorSpec(low=0., high=1., shape=torch.Size((batch_size, 3))).rand()

    def training_step(self, batch, batch_idx: int):
        scheduler_actor, scheduler_critic, scheduler_alpha, scheduler_backbone = self.lr_schedulers()
        scheduler_actor.step()
        scheduler_critic.step()
        scheduler_alpha.step()
        scheduler_backbone.step()

        env_state: SharedMemory
        game_idx: int
        env_state, game_idx = batch
        is_done = env_state.is_done

        next_state, step, backbone_loss = self.forward_game_state(env_state, with_loss_and_grad=is_done)
        if not is_done:
            # calculate action and submit it.
            if self.global_step * 3 < self.init_random_batches:
                next_action = self.random_action(next_state.batch_size[0]).to(self.device)
            else:
                next_action = self.forward_action(next_state, exploration_type=ExplorationType.RANDOM)
            next_state['action'] = next_action
            env_state.set_action(next_action.detach())
        else:
            # last step in env - log final score.
            score = next_state['score'].mean().item()
            self.log(name='train/mae_rmse', value=-score, on_step=True, on_epoch=True,
                     batch_size=env_state.current_batch_size)

        next_state = next_state.detach()
        if step == 0:
            # first step, no previous state available. Store state and finish.
            self.game_state[game_idx] = next_state
            return

        # previous state available.
        state: Optional[TensorDict] = self.game_state[game_idx]
        assert state is not None

        state['next'] = next_state
        if not is_done:
            self.game_state[game_idx] = next_state
        else:
            self.game_state[game_idx] = None

        # calculate reward.
        state['next', 'reward'] = state['next', 'score'] - state['score']

        self.replay_buffer.extend(state)

        optimizer_actor, optimizer_critic, optimizer_alpha, optimizer_backbone = self.optimizers()

        if len(self.replay_buffer) > self.init_random_batches * self.batch_size:
            actor_losses = []
            critic_losses = []
            alpha_losses = []

            for iter_idx in range(self.rl_iters_per_step):
                subdata = self.replay_buffer.sample()
                subdata = subdata.to(self.device)
                loss_td = self.rl_loss_module(subdata)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                # Update actor
                optimizer_actor.zero_grad()
                self.manual_backward(actor_loss)
                optimizer_actor.step()
                actor_losses.append(actor_loss.mean().item())

                # Update critic
                optimizer_critic.zero_grad()
                self.manual_backward(q_loss)
                optimizer_critic.step()
                critic_losses.append(q_loss.mean().item())

                # Update alpha
                optimizer_alpha.zero_grad()
                self.manual_backward(alpha_loss)
                optimizer_alpha.step()
                alpha_losses.append(alpha_loss.mean().item())

                self.target_net_updater.step()

            self.log(name='train/actor_loss', value=sum(actor_losses) / len(actor_losses), on_step=True, on_epoch=False,
                     batch_size=env_state.current_batch_size)
            self.log(name='train/critic_loss', value=sum(critic_losses) / len(critic_losses), on_step=True,
                     on_epoch=False, batch_size=env_state.current_batch_size)
            self.log(name='train/alpha_loss', value=sum(alpha_losses) / len(alpha_losses), on_step=True, on_epoch=False,
                     batch_size=env_state.current_batch_size)

        if self.global_step > self.init_backbone_batches * 3:
            if is_done:
                optimizer_backbone.zero_grad()
                self.manual_backward(backbone_loss)
                optimizer_backbone.step()
                self.log(name='train/mae_loss', value=backbone_loss.mean().item(), on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        env_state: SharedMemory
        game_idx: int
        env_state, game_idx = batch
        is_done = env_state.is_done

        next_state, step, backbone_loss = self.forward_game_state(env_state, with_loss_and_grad=False)
        score = next_state['score'].mean().item()

        if is_done:
            self.log(name='val/mae_rmse', value=-score, on_step=True, on_epoch=True,
                     batch_size=env_state.current_batch_size)
        else:
            next_action = self.forward_action(next_state, exploration_type=ExplorationType.MEAN)
            env_state.set_action(next_action.detach())
