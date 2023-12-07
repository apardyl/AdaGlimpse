import sys
from typing import Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SACLoss, SoftUpdate

from architectures.glimpse_selectors import ElasticAttentionMapEntropy
from architectures.mae import MaskedAutoencoderViT, mae_vit_base_patch16
from architectures.rl.actor_critic import ActorCritic
from architectures.rl.glimpse_game import GlimpseGameModelWrapper, GlimpseGameEnv, GlimpseGameDataCollector
from architectures.utils import MetricMixin, RevNormalizer
from datasets.base import BaseDataModule


class MaeWrapper(GlimpseGameModelWrapper):

    def __init__(self, mae):
        super().__init__()
        self.mae = mae
        self.extractor = ElasticAttentionMapEntropy(self)
        self.rev_normalizer = None

    def _calculate_loss(self, out, images):
        if self.rev_normalizer is None:
            self.rev_normalizer = RevNormalizer(images.device)

        pred = self.mae.unpatchify(out)
        pred = self.rev_normalizer(pred)
        target = self.rev_normalizer(images)
        return torch.sqrt(
            torch.nn.functional.mse_loss(pred, target, reduce=False)
            .reshape(pred.shape[0], -1)
            .mean(dim=-1, keepdim=True)
        )

    def __call__(self, images, patches, coords
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.mae.eval()
        with torch.no_grad():
            latent = self.mae.forward_encoder(patches, coords=coords)
            out = self.mae.forward_decoder(latent)
            loss = self._calculate_loss(out, images)
            state = self.extractor().reshape(images.shape[0], -1)
            return state, loss


class RlMAE(LightningModule, MetricMixin):
    rl_state_dim = 196
    rl_hidden_dim = 256
    rl_action_dim = 3

    def __init__(self, datamodule: BaseDataModule, pretrained_mae_path=None, num_glimpses=12,
                 rl_iters_per_step=1, batch_size=32, epochs=10, init_random_batches=100, init_backbone_batches=50000,
                 rl_batch_size=256, replay_buffer_size=10000, lr=3e-4, backbone_lr=1e-5) -> None:
        super().__init__()

        self.steps_per_epoch = None
        self.num_glimpses = num_glimpses
        self.rl_iters_per_step = rl_iters_per_step
        self.batch_size = batch_size
        self.rl_batch_size = rl_batch_size
        self.epochs = epochs
        self.init_random_batches = init_random_batches
        self.init_backbone_batches = init_backbone_batches
        self.lr = lr
        self.backbone_lr = backbone_lr

        self.datamodule = datamodule

        self.automatic_optimization = False  # disable lightning automation

        self.mae = mae_vit_base_patch16(img_size=datamodule.image_size, out_chans=3)
        # noinspection PyTypeChecker
        self.mae: MaskedAutoencoderViT = torch.compile(self.mae, mode='reduce-overhead')

        if pretrained_mae_path:
            self.load_pretrained_elastic(pretrained_mae_path)

        self.actor_critic = ActorCritic(
            self.rl_state_dim,
            self.rl_hidden_dim,
            self.rl_action_dim,
            self.num_glimpses
        )
        self.rl_loss_module = SACLoss(actor_network=self.actor_critic.policy_module,
                                      qvalue_network=self.actor_critic.qvalue_module,
                                      loss_function='smooth_l1',
                                      delay_actor=False,
                                      delay_qvalue=True,
                                      alpha_init=1.0)
        self.rl_loss_module.make_value_estimator(gamma=0.99)
        self.target_net_updater = SoftUpdate(self.rl_loss_module, eps=0.995)

        self.env_train = None
        self.env_val = None
        self.train_collector = None

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=replay_buffer_size
            ),
            prefetch=3,
            batch_size=self.rl_batch_size
        )

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
        return GlimpseGameDataCollector(
            self.env_train,
            self.actor_critic.policy_module,
            frames_per_batch=self.batch_size,
            init_random_frames=self.init_random_batches * self.rl_batch_size,
            device=self.device
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.env_val.iterate_with_policy(self.actor_critic.policy_module)

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: str) -> None:
        if stage != 'fit':
            raise NotImplemented()

        self.datamodule.setup(stage)

        self.env_train = GlimpseGameEnv(
            glimpse_model=MaeWrapper(self.mae),
            dataloader=self.datamodule.train_dataloader(),
            num_glimpses=self.num_glimpses,
            batch_size=self.batch_size,
            state_dim=self.rl_state_dim,
            device=self.device
        )

        self.steps_per_epoch = len(self.env_train)

        self.env_val = GlimpseGameEnv(
            glimpse_model=MaeWrapper(self.mae),
            dataloader=self.datamodule.val_dataloader(),
            num_glimpses=self.num_glimpses,
            batch_size=self.batch_size,
            state_dim=self.rl_state_dim,
            device=self.device
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        if not hasattr(self.rl_loss_module, '_cache'):
            self.rl_loss_module._cache = {}

    def training_step(self, tensordict_data, batch_idx):
        scheduler_actor, scheduler_critic, scheduler_alpha, scheduler_backbone = self.lr_schedulers()
        scheduler_actor.step()
        scheduler_critic.step()
        scheduler_alpha.step()
        scheduler_backbone.step()

        data_view = tensordict_data.reshape(-1)
        self.replay_buffer.extend(data_view.cpu())

        optimizer_actor, optimizer_critic, optimizer_alpha, optimizer_backbone = self.optimizers()

        self.log(name='train/mae_rmse', value=self.env_train.last_final_rmse, on_step=True, on_epoch=False)

        if len(self.replay_buffer) > self.init_random_batches * self.batch_size:
            actor_losses = []
            critic_losses = []
            alpha_losses = []

            for iter_idx in range(self.rl_iters_per_step):
                subdata = self.replay_buffer.sample()
                subdata = subdata.to(tensordict_data.device)
                loss_td = self.rl_loss_module(subdata)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                # Update actor
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()
                actor_losses.append(actor_loss.mean().item())

                # Update critic
                optimizer_critic.zero_grad()
                q_loss.backward()
                optimizer_critic.step()
                critic_losses.append(q_loss.mean().item())

                # Update alpha
                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()
                alpha_losses.append(alpha_loss.mean().item())

                self.target_net_updater.step()

            self.log(name='train/actor_loss', value=sum(actor_losses) / len(actor_losses), on_step=True, on_epoch=False)
            self.log(name='train/critic_loss', value=sum(critic_losses) / len(critic_losses), on_step=True,
                     on_epoch=False)
            self.log(name='train/alpha_loss', value=sum(alpha_losses) / len(alpha_losses), on_step=True, on_epoch=False)

        if self.global_step > self.init_backbone_batches * 3:
            final_sampler = self.env_train.pop_prev_sampler()
            if final_sampler is not None:
                self.mae.train()
                latent = self.mae.forward_encoder(final_sampler.patches, coords=final_sampler.coords)
                out = self.mae.forward_decoder(latent)
                rec_loss = self.mae.forward_reconstruction_loss(final_sampler.images, out)
                optimizer_backbone.zero_grad()
                rec_loss.backward()
                optimizer_backbone.step()
                mae_loss = rec_loss.mean().item()

                self.log(name='train/mae_loss', value=mae_loss, on_step=True, on_epoch=False)

    def validation_step(self, eval_rollout):
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            mean_mae_loss = eval_rollout['mae_loss'][:, -1].mean().item()
            self.log(name='val/mae_loss', value=mean_mae_loss, on_step=True, on_epoch=True)
            self.log(name='val/mae_rmse', value=self.env_val.last_final_rmse, on_step=True, on_epoch=False)
