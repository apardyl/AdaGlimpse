import argparse
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import partial
from typing import Optional, Dict, Any, Tuple, Union

import torch
import torchmetrics
from lightning.pytorch.strategies import ParallelStrategy
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tensordict import TensorDict
from torch.utils.data import DistributedSampler
from torchrl.data import ReplayBuffer, LazyTensorStorage, BoundedTensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate, SACLoss
from torchvision.models.segmentation import DeepLabV3, deeplabv3_resnet101

from architectures.base import AutoconfigLightningModule
from architectures.mae import MaskedAutoencoderViT, mae_vit_base_patch16, mae_vit_small_patch16, mae_vit_large_patch16
from architectures.rl.actor_critic import ActorCritic
from architectures.rl.glimpse_engine import glimpse_engine, BaseGlimpseEngine
from architectures.rl.shared_memory import SharedMemory
from architectures.utils import MetricMixin, RevNormalizer, filter_checkpoint
from datasets.base import BaseDataModule
from datasets.classification import BaseClassificationDataModule
from datasets.patch_sampler import PretrainingSampler
from datasets.segmentation import BaseSegmentationDataModule


class BaseRlMAE(AutoconfigLightningModule, MetricMixin, ABC):
    internal_data = True
    checkpoint_metric = None
    autograd_backbone = True
    decoder_type = 'none'
    decoder_out_channels = 3

    def __init__(self, datamodule: BaseDataModule, backbone_size='base', pretrained_mae_path=None, num_glimpses=14,
                 max_glimpse_size_ratio=1.0, glimpse_grid_size=2, rl_iters_per_step=1, epochs=100,
                 init_random_batches=1000, freeze_backbone_epochs=10, rl_batch_size=128, replay_buffer_size=10000,
                 lr=3e-4, backbone_lr=1e-5, parallel_games=2, backbone_training_type: str = 'constant',
                 rl_loss_function: str = 'smooth_l1', glimpse_size_penalty: float = 0., reward_type='diff',
                 early_stop_threshold=None, extract_latent_layer=None, rl_target_entropy=None,
                 teacher_type=None, teacher_path=None, pretraining=False, pretrained_checkpoint=None,
                 exclude_rl_inputs=None, teacher_pretraining=False, backbone_decay=1e-4, **_) -> None:
        super().__init__()

        self.steps_per_epoch = None
        self.num_glimpses = num_glimpses
        self.max_glimpse_size_ratio = max_glimpse_size_ratio
        self.glimpse_grid_size = glimpse_grid_size
        self.rl_iters_per_step = rl_iters_per_step
        self.rl_batch_size = rl_batch_size
        self.epochs = epochs
        self.pretraining = pretraining
        self.init_random_batches = init_random_batches
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.lr = lr
        self.backbone_lr = backbone_lr
        self.backbone_training_type = backbone_training_type
        self.glimpse_size_penalty = glimpse_size_penalty
        self.reward_type = reward_type
        self.early_stop_threshold = early_stop_threshold
        self.extract_latent_layer = extract_latent_layer
        self.rl_target_entropy = rl_target_entropy
        self.teacher_type = teacher_type
        self.teacher_path = teacher_path
        self.teacher_pretraining = teacher_pretraining
        self.backbone_decay = backbone_decay

        self.replay_buffer_size = replay_buffer_size
        self.parallel_games = parallel_games

        self.datamodule = datamodule

        self.automatic_optimization = False  # disable lightning automation

        self.mae = None
        if not self.teacher_pretraining:
            self.mae = {
                'small': mae_vit_small_patch16,
                'base': mae_vit_base_patch16,
                'large': mae_vit_large_patch16
            }[backbone_size](img_size=datamodule.image_size, out_chans=self.decoder_out_channels,
                             decoder_type=self.decoder_type)
            # noinspection PyTypeChecker
            self.mae: MaskedAutoencoderViT = torch.compile(self.mae, mode='reduce-overhead')

        self.actor_critic = ActorCritic(embed_dim=self.mae.patch_embed.embed_dim if self.mae is not None else 1,
                                        patch_num=self.num_glimpses * (self.glimpse_grid_size ** 2),
                                        exclude_inputs=exclude_rl_inputs)

        if pretrained_mae_path is not None:
            self.load_pretrained_elastic(pretrained_mae_path)

        if self.rl_target_entropy is None:
            self.rl_target_entropy = 'auto'

        self.rl_loss_module = SACLoss(actor_network=self.actor_critic.policy_module,
                                      qvalue_network=self.actor_critic.qvalue_module,
                                      loss_function=rl_loss_function,
                                      delay_actor=False,
                                      delay_qvalue=True,
                                      alpha_init=1.0,
                                      target_entropy=self.rl_target_entropy, )
        self.rl_loss_module.make_value_estimator(
            gamma=0.99,
            average_rewards=self.reward_type == 'autonorm'
        )
        self.target_net_updater = SoftUpdate(self.rl_loss_module, eps=0.995)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.engine: Optional[BaseGlimpseEngine] = None
        self.replay_buffer = None
        self.game_state = [None] * max(self.parallel_games, 1)
        self.add_pos_embed = False
        self.distillation_targets = [None] * max(self.parallel_games, 1)

        self.save_hyperparameters(ignore=['datamodule'])
        self._user_forward_hook = None

        self.real_train_step = 0

        self.teacher_model: Optional[Union[MaskedAutoencoderViT, DeepLabV3]] = None
        if self.teacher_type == 'vit':
            self.teacher_model = mae_vit_base_patch16(img_size=datamodule.image_size, decoder_type='none')
        elif self.teacher_type == 'deeplab':
            self.teacher_model = deeplabv3_resnet101(num_classes=self.decoder_out_channels)
        if self.teacher_path is not None and not self.teacher_pretraining:
            self.load_teacher(teacher_path)
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        self.pretraining_sampler = PretrainingSampler(max_glimpses=self.num_glimpses,
                                                      glimpse_size=self.glimpse_grid_size)

        if pretrained_checkpoint is not None:
            self.load_pretrained(pretrained_checkpoint)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(BaseRlMAE.__name__)
        parser.add_argument('--lr',
                            help='learning-rate',
                            type=float,
                            default=1e-3)
        parser.add_argument('--backbone-size',
                            help='backbone ViT size',
                            type=str,
                            default='base')
        parser.add_argument('--backbone-lr',
                            help='backbone learning-rate',
                            type=float,
                            default=1e-5)
        parser.add_argument('--pretrained-mae-path',
                            help='path to pretrained MAE weights',
                            type=str,
                            default=None)
        parser.add_argument("--pretrained-checkpoint",
                            help='path to a saved model state to load (only loads weights, does not resume training)',
                            type=str,
                            default=None)
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
                            default=10000)
        parser.add_argument('--freeze-backbone-epochs',
                            help='number of rl training epochs before starting to train the backbone',
                            type=int,
                            default=10)
        parser.add_argument('--backbone-training-type',
                            help='type of backbone training regime',
                            choices=['disabled', 'constant', 'alternating'],
                            default='alternating',
                            type=str)
        parser.add_argument('--rl-loss-function',
                            help='type of loss function for rl training',
                            default='l2',
                            type=str)
        parser.add_argument('--rl-batch-size',
                            help='batch size of the rl loop',
                            type=int,
                            default=128)
        parser.add_argument('--replay-buffer-size',
                            help='rl replay buffer size in episodes',
                            type=int,
                            default=10000)
        parser.add_argument('--parallel-games',
                            help='number of parallel game workers (0 for single-threaded)',
                            type=int,
                            default=2)
        parser.add_argument('--max-glimpse-size-ratio',
                            help='maximum glimpse size relative to full image size',
                            type=float,
                            default=1.0)
        parser.add_argument('--glimpse-grid-size',
                            help='size of glimpse sampling grid in patches along grid side',
                            type=int,
                            default=2)
        parser.add_argument('--glimpse-size-penalty',
                            help='penalty coefficient for large glimpses',
                            type=float,
                            default=0)
        parser.add_argument('--reward-type',
                            help='reward type selection',
                            type=str,
                            default='diff',
                            choices=['diff', 'simple', 'autonorm'])
        parser.add_argument('--early-stop-threshold',
                            help='exploration early stop score threshold',
                            type=float,
                            default=None)
        parser.add_argument('--extract-latent-layer',
                            help='number of encoder layer to extract latent for rl from (default = last)',
                            type=int,
                            default=None)
        parser.add_argument('--rl-target-entropy',
                            help='target entropy for SAC algorithm',
                            type=float,
                            default=None)
        parser.add_argument('--teacher-type',
                            help='teacher model type',
                            type=str,
                            default=None,
                            choices=['vit', 'deeplab'])
        parser.add_argument('--teacher-path',
                            help='use distilled targets by running a model on full image',
                            type=str,
                            default=None)
        parser.add_argument('--pretraining',
                            help='backbone pretraining mode (random action, disables rl)',
                            type=bool,
                            default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--teacher-pretraining',
                            help='teacher pretraining mode',
                            type=bool,
                            default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--exclude-rl-inputs',
                            help='exclude parts of the rl state for ablation study',
                            type=str,
                            action='extend',
                            nargs='*')
        parser.add_argument('--backbone-decay',
                            help='backbone weight decay',
                            type=float,
                            default=1e-4)
        return parent_parser

    def configure_optimizers(self):
        critic_params = list(self.rl_loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(self.rl_loss_module.actor_network_params.flatten_keys().values())

        actor_optimizer = torch.optim.AdamW(actor_params, self.lr)
        actor_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=actor_optimizer,
            max_lr=self.lr,
            pct_start=0.01,
            div_factor=1,
            final_div_factor=1000,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        critic_optimizer = torch.optim.AdamW(critic_params, self.lr)
        critic_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=critic_optimizer,
            max_lr=self.lr,
            pct_start=0.01,
            div_factor=1,
            final_div_factor=1000,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        alpha_optimizer = torch.optim.AdamW([self.rl_loss_module.log_alpha], self.lr)
        alpha_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=alpha_optimizer,
            max_lr=self.lr,
            pct_start=0.01,
            div_factor=1,
            final_div_factor=1000,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        backbone_params = self.mae.parameters() if not self.teacher_pretraining else self.teacher_model.parameters()
        backbone_optimizer = torch.optim.AdamW(backbone_params, self.lr, weight_decay=self.backbone_decay)
        backbone_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=backbone_optimizer,
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
                    'interval': 'step',
                    'name': 'lr/actor'
                }
            },
            {
                "optimizer": critic_optimizer,
                "lr_scheduler": {
                    'scheduler': critic_scheduler,
                    'interval': 'step',
                    'name': 'lr/critic'
                }
            },
            {
                "optimizer": alpha_optimizer,
                "lr_scheduler": {
                    'scheduler': alpha_scheduler,
                    'interval': 'step',
                    'name': 'lr/alpha'
                }
            },
            {
                "optimizer": backbone_optimizer,
                "lr_scheduler": {
                    'scheduler': backbone_scheduler,
                    'interval': 'step',
                    'name': 'lr/backbone'
                }
            }
        )

    def load_pretrained_elastic(self, path=""):
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint["state_dict"]
            checkpoint = {k[4:]: v for k, v in checkpoint.items() if k.startswith('mae.')}
        elif 'model' in checkpoint:
            checkpoint = checkpoint["model"]
            checkpoint = {'_orig_mod.' + k: v for k, v in checkpoint.items()}
        else:
            raise ValueError("Unable to parse pretrained model checkpoint")
        del checkpoint['_orig_mod.pos_embed']
        if (hasattr(self.mae, 'decoder_pred') and '_orig_mod.decoder_pred.weight' in checkpoint
                and checkpoint['_orig_mod.decoder_pred.weight'].shape != self.mae.decoder_pred.weight.shape):
            del checkpoint['_orig_mod.decoder_pred.weight']
            del checkpoint['_orig_mod.decoder_pred.bias']
        print(self.mae.load_state_dict(checkpoint, strict=False), file=sys.stderr)

    def load_pretrained(self, path=""):
        checkpoint = torch.load(path, map_location='cpu')
        checkpoint = checkpoint["state_dict"]
        self.actor_critic.load_state_dict(filter_checkpoint(checkpoint, 'actor_critic.'), strict=True)
        self._restore_rl(checkpoint)
        if (hasattr(self.mae, 'decoder_pred') and 'mae._orig_mod.decoder_pred.weight' in checkpoint
                and checkpoint['mae._orig_mod.decoder_pred.weight'].shape != self.mae.decoder_pred.weight.shape):
            del checkpoint['mae._orig_mod.decoder_pred.weight']
            del checkpoint['mae._orig_mod.decoder_pred.bias']
        print(self.mae.load_state_dict(filter_checkpoint(checkpoint, 'mae.'), strict=False))

    def load_teacher(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        if self.teacher_type == 'deeplab':
            print(self.teacher_model.load_state_dict(
                filter_checkpoint(checkpoint['state_dict'], 'teacher_model.'), strict=False),
                file=sys.stderr)
        elif self.teacher_type == 'vit':
            print(self.teacher_model.load_state_dict(checkpoint['model'], strict=False), file=sys.stderr)
        else:
            raise NotImplementedError()

    @staticmethod
    def _copy_target_tensor_fn(target: torch.Tensor, batch: Dict[str, torch.Tensor]):
        pass

    @staticmethod
    def _create_target_tensor_fn(batch_size: int, image_size: Tuple[int, int]):
        return torch.zeros((batch_size, 0))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.pretraining or self.teacher_pretraining:
            return self.train_loader
        return self.engine.get_loader(dataloader=self.train_loader)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.pretraining or self.teacher_pretraining:
            return self.val_loader
        return self.engine.get_loader(dataloader=self.val_loader)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.engine.get_loader(dataloader=self.test_loader)

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: str) -> None:
        if stage != 'fit' and stage != 'validate' and stage != 'test':
            raise NotImplementedError()

        self.datamodule.setup(stage)

        if stage == 'fit':
            if isinstance(self.trainer.strategy, ParallelStrategy):
                self.train_loader = self.datamodule.train_dataloader(
                    sampler=DistributedSampler(self.datamodule.train_dataset))
            else:
                self.train_loader = self.datamodule.train_dataloader()

            if self.pretraining or self.teacher_pretraining:
                self.steps_per_epoch = len(self.train_loader)
            else:
                self.steps_per_epoch = len(self.train_loader) * (self.num_glimpses + 1)

            self.replay_buffer = ReplayBuffer(
                storage=LazyTensorStorage(
                    max_size=self.replay_buffer_size,
                    device=self.device
                ),
                batch_size=self.rl_batch_size
            )

        if stage == 'fit' or stage == 'validate':
            if isinstance(self.trainer.strategy, ParallelStrategy):
                self.val_loader = self.datamodule.val_dataloader(
                    sampler=DistributedSampler(self.datamodule.val_dataset))
            else:
                self.val_loader = self.datamodule.val_dataloader()
            if stage == 'fit':
                self.datamodule.train_dataset.set_patch_sampler(
                    self.pretraining_sampler if getattr(self, 'pretraining', False) else None)
            self.datamodule.val_dataset.set_patch_sampler(
                self.pretraining_sampler if getattr(self, 'pretraining', False) else None)

        if stage == 'test':
            self.test_loader = self.datamodule.test_dataloader()

        self.engine = glimpse_engine(
            max_glimpses=self.num_glimpses,
            glimpse_grid_size=self.glimpse_grid_size,
            native_patch_size=(16, 16),
            batch_size=max(self.datamodule.train_batch_size, self.datamodule.eval_batch_size),
            max_glimpse_size_ratio=self.max_glimpse_size_ratio,
            device=self.device,
            image_size=self.datamodule.image_size,
            num_parallel_games=self.parallel_games,
            create_target_tensor_fn=self._create_target_tensor_fn,
            copy_target_tensor_fn=self._copy_target_tensor_fn,
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        # fix for torch rl removing caches on copy to device.
        if not hasattr(self.rl_loss_module, '_cache'):
            self.rl_loss_module._cache = {}

        if isinstance(self.trainer.strategy, ParallelStrategy):
            self.train_loader.sampler.set_epoch(self.trainer.current_epoch)

        self.replay_buffer.empty()

    @abstractmethod
    def _forward_task(self, images: torch.Tensor, targets: torch.Tensor, latent: torch.Tensor, is_done: bool,
                      with_loss_and_grad: bool, mode: str, distilled_target: Optional[torch.Tensor] = None):
        """This function implements the task-specific forward pass of the model, including computing the loss value,
        score for RL training, as well as logging any task-specific metrics."""
        raise NotImplementedError()

    def forward_game_state(self, env_state: SharedMemory, mode: str, distilled_target: Optional[torch.Tensor] = None):
        is_done = env_state.is_done
        with_loss_and_grad = mode == 'train' and is_done
        with (nullcontext() if self.autograd_backbone and with_loss_and_grad else torch.no_grad()):
            step = env_state.current_glimpse
            latent, pos_embed = self.mae.forward_encoder(env_state.current_patches, coords=env_state.current_coords,
                                                         pad_mask=env_state.current_mask,
                                                         aux_latent_layer=self.extract_latent_layer)

            out, loss, score = self._forward_task(env_state.images, env_state.target, latent, is_done,
                                                  with_loss_and_grad, mode, distilled_target)

            if self.extract_latent_layer is not None:
                latent = self.mae.aux_latent

            all_coords = env_state.all_coords.clone()
            observation = torch.zeros(latent.shape[0], all_coords.shape[1] + 1,
                                      latent.shape[-1], device=latent.device, dtype=latent.dtype)
            observation[:, :latent.shape[1]].copy_(latent)

            attention = torch.zeros(all_coords.shape[0], all_coords.shape[1], 1, device=latent.device,
                                    dtype=latent.dtype)
            attention[:, :latent.shape[1] - 1].copy_(self.mae.encoder_attention_rollout())

            if self.add_pos_embed:
                observation[:, 1:latent.shape[1]].add_(pos_embed)

            observation.detach_()

            if self.early_stop_threshold is not None:
                if not is_done:
                    env_state.done = torch.logical_or(env_state.done, torch.ge(score, self.early_stop_threshold))
                else:
                    avg_steps_taken = (env_state.all_mask.shape[1] - env_state.all_mask.sum(dim=1)
                                       ).mean() / self.glimpse_grid_size ** 2
                    self.log(name=f'{mode}/avg_steps', value=avg_steps_taken.item(), on_step=True,
                             on_epoch=True, batch_size=score.shape[0])

        done = env_state.done.clone()

        next_state = TensorDict({
            'observation': observation,
            'mask': env_state.all_mask.clone(),
            'coords': all_coords,
            'step': torch.ones(size=(latent.shape[0], 1), dtype=torch.long, device=latent.device) * step,
            'done': done,
            'terminated': done,
            'score': score,
            'attention': attention,
            'patches': env_state.all_patches.clone()
        }, batch_size=observation.shape[0])

        self.call_user_forward_hook(env_state, out, score)

        return next_state, step, loss

    @torch.no_grad()
    def forward_action(self, state_dict: TensorDict, exploration_type: ExplorationType):
        with set_exploration_type(exploration_type):
            action = self.actor_critic.policy_module(state_dict)['action']
            return action

    @staticmethod
    def random_action(batch_size):
        return BoundedTensorSpec(low=0., high=1., shape=torch.Size((batch_size, 3))).rand()

    @property
    def is_rl_training_enabled(self):
        if self.freeze_backbone_epochs > self.current_epoch:
            return True
        if self.backbone_training_type == 'alternating':
            return self.current_epoch % 2 == 0
        return True

    @property
    def is_backbone_training_enabled(self):
        if self.freeze_backbone_epochs > self.current_epoch:
            return False
        if self.backbone_training_type == 'disabled':
            return False
        elif self.backbone_training_type == 'constant':
            return True
        elif self.backbone_training_type == 'alternating':
            return self.current_epoch % 2 == 1
        else:
            raise ValueError('Unknown backbone training mode')

    def rl_training_step(self, optimizer_actor, optimizer_critic, optimizer_alpha, batch_size):
        actor_losses = []
        critic_losses = []
        alpha_losses = []

        for iter_idx in range(self.rl_iters_per_step):
            rl_batch = self.replay_buffer.sample()
            rl_batch = rl_batch.to(self.device)
            loss_td = self.rl_loss_module(rl_batch)

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
                 batch_size=batch_size)
        self.log(name='train/critic_loss', value=sum(critic_losses) / len(critic_losses), on_step=True,
                 on_epoch=False, batch_size=batch_size)
        self.log(name='train/alpha_loss', value=sum(alpha_losses) / len(alpha_losses), on_step=True, on_epoch=False,
                 batch_size=batch_size)

    def backbone_training_step(self, optimizer_backbone, backbone_loss):
        optimizer_backbone.zero_grad()
        self.manual_backward(backbone_loss)
        optimizer_backbone.step()
        self.log(name='train/backbone_loss', value=backbone_loss.mean().item(), on_step=True, on_epoch=False)

    @torch.inference_mode()
    def teacher_step(self, images: torch.Tensor, game_idx: int):
        if self.teacher_type == 'vit':
            latent, _ = self.teacher_model.forward_encoder(images)
            out = self.teacher_model.forward_head(latent)
            self.distillation_targets[game_idx] = out.clone().detach()
        elif self.teacher_type == 'deeplab':
            out = self.teacher_model(images)['out']
            self.distillation_targets[game_idx] = out.clone().detach()

    def teacher_training_step(self, batch, mode):
        raise NotImplementedError()

    def pretraining_step(self, batch, mode):
        if self.teacher_pretraining:
            self.teacher_training_step(batch, mode)
        else:
            if mode == 'train' and self.teacher_model is not None:
                self.teacher_step(batch['image'], 0)
            latent, pos_embed = self.mae.forward_encoder(batch['patches'], coords=batch['coords'])
            _, backbone_loss, _ = self._forward_task(batch['image'], batch.get('mask', None), latent,
                                                     is_done=True, with_loss_and_grad=True, mode=mode,
                                                     distilled_target=self.distillation_targets[
                                                         0] if mode == 'train' else None)
            if mode == 'train':
                optimizer_actor, optimizer_critic, optimizer_alpha, optimizer_backbone = self.optimizers()
                self.backbone_training_step(optimizer_backbone, backbone_loss)
                _, _, _, scheduler_backbone = self.lr_schedulers()
                scheduler_backbone.step()

    def training_step(self, batch, batch_idx: int):
        if self.pretraining or self.teacher_pretraining:
            self.pretraining_step(batch, 'train')
            return

        scheduler_actor, scheduler_critic, scheduler_alpha, scheduler_backbone = self.lr_schedulers()
        scheduler_actor.step()
        scheduler_critic.step()
        scheduler_alpha.step()
        scheduler_backbone.step()
        self.real_train_step += 1

        env_state: SharedMemory
        game_idx: int
        env_state, game_idx = batch

        if self.teacher_model is not None and env_state.current_glimpse == 0:
            self.teacher_step(env_state.images, game_idx)

        next_state, step, backbone_loss = self.forward_game_state(env_state, mode='train',
                                                                  distilled_target=self.distillation_targets[game_idx])
        if not env_state.is_done:
            # calculate action and submit it.
            if self.real_train_step < self.init_random_batches:
                next_action = self.random_action(next_state.batch_size[0]).to(self.device)
            else:
                next_action = self.forward_action(next_state, exploration_type=ExplorationType.RANDOM)
            next_state['action'] = next_action
            env_state.action = next_action.detach()

        is_done = env_state.is_done

        if self.is_rl_training_enabled:
            next_state = next_state.detach()

            if step == 0:
                # first step, no previous state available. Store state and finish.
                self.game_state[game_idx] = next_state
                return

            # previous state available.
            state: Optional[TensorDict] = self.game_state[game_idx]
            assert state is not None

            state['next'] = next_state

            active_mask = torch.logical_not(torch.logical_and(state['done'], state['next', 'done'])).squeeze(1)

            if not is_done:
                self.game_state[game_idx] = next_state
            else:
                self.game_state[game_idx] = None
                self.log(name='train/final_score',
                         value=state['next', 'score'].sum().item() / state['next', 'score'].shape[0], on_step=True,
                         on_epoch=False, batch_size=state['next', 'score'].shape[0])

            # calculate reward.
            if self.reward_type == 'simple' or self.reward_type == 'autonorm':
                state['next', 'reward'] = state['next', 'score']
            elif self.reward_type == 'diff':
                state['next', 'reward'] = state['next', 'score'] - state['score']
            else:
                assert False

            if not is_done and self.glimpse_size_penalty > 0.:
                state['next', 'reward'] = (state['next', 'reward'] -
                                           state['next', 'action'][:, -1:] * self.glimpse_size_penalty)

            self.replay_buffer.extend(state[active_mask])

        optimizer_actor, optimizer_critic, optimizer_alpha, optimizer_backbone = self.optimizers()

        if self.is_rl_training_enabled and len(self.replay_buffer) >= self.rl_batch_size:
            self.rl_training_step(optimizer_actor, optimizer_critic, optimizer_alpha, env_state.current_batch_size)

        if self.is_backbone_training_enabled and is_done:
            self.backbone_training_step(optimizer_backbone, backbone_loss)

    def inference_step(self, batch, batch_idx, mode):
        if self.pretraining or self.teacher_pretraining:
            self.pretraining_step(batch, 'val')
            return

        env_state: SharedMemory
        game_idx: int
        env_state, game_idx = batch

        next_state, step, backbone_loss = self.forward_game_state(env_state, mode=mode)

        if not env_state.is_done:
            next_action = self.forward_action(next_state, exploration_type=ExplorationType.MEAN)
            env_state.action = next_action.detach()

    def validation_step(self, batch, batch_idx):
        self.inference_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.inference_step(batch, batch_idx, 'test')

    @property
    def user_forward_hook(self):
        return self._user_forward_hook

    @user_forward_hook.setter
    def user_forward_hook(self, hook):
        self._user_forward_hook = hook

    @user_forward_hook.deleter
    def user_forward_hook(self):
        self._user_forward_hook = None

    def call_user_forward_hook(self, *args, **kwargs):
        if self._user_forward_hook:
            with torch.no_grad():
                self._user_forward_hook(*args, **kwargs)

    def _restore_rl(self, state_dict):
        rl_state_dict = filter_checkpoint(state_dict, 'rl_loss_module.')
        print(self.rl_loss_module.load_state_dict(rl_state_dict, strict=False))

        def stub(*args, **kwargs):
            pass

        # prevent auto-restore by lightning
        self.rl_loss_module._load_from_state_dict = stub
        self.rl_loss_module.actor_network_params._load_from_state_dict = stub
        self.rl_loss_module.qvalue_network_params._load_from_state_dict = stub
        self.rl_loss_module.target_qvalue_network_params._load_from_state_dict = stub

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # manual restore RL state (workaround issue in torchrl)
        state_dict = checkpoint['state_dict']
        self._restore_rl(state_dict)

        self.real_train_step = checkpoint.get('real_train_step', 0)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['real_train_step'] = self.real_train_step


class ReconstructionRlMAE(BaseRlMAE):
    checkpoint_metric = 'val/rmse'
    decoder_type = 'standard'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.rev_normalizer = RevNormalizer()
        self.define_metric('rmse', partial(torchmetrics.MeanSquaredError, squared=False))

    def _forward_task(self, images: torch.Tensor, targets: torch.Tensor, latent: torch.Tensor, is_done: bool,
                      with_loss_and_grad: bool, mode: str, distilled_target: Optional[torch.Tensor] = None):
        out = self.mae.forward_decoder(latent)
        loss = None
        if with_loss_and_grad:
            loss = self.mae.forward_reconstruction_loss(images, out)

        pred = self.mae.unpatchify(out)
        pred = self.rev_normalizer(pred)
        target = self.rev_normalizer(images)
        score = -torch.sqrt(
            torch.nn.functional.mse_loss(pred, target, reduce=False)
            .reshape(pred.shape[0], -1)
            .mean(dim=-1, keepdim=True)
        )

        if is_done:
            self.log_metric(mode, 'rmse', pred.detach(), target, on_epoch=True, batch_size=latent.shape[0])

        return out, loss, score


class ClassificationRlMAE(BaseRlMAE):
    checkpoint_metric = 'val/accuracy'
    checkpoint_metric_mode = 'max'
    decoder_type = 'none'

    # autograd_backbone = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert isinstance(self.datamodule, BaseClassificationDataModule)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.define_metric('accuracy',
                           partial(torchmetrics.classification.MulticlassAccuracy,
                                   num_classes=self.datamodule.cls_num_classes,
                                   average='micro'))
        self.define_metric('teacher_accuracy',
                           partial(torchmetrics.classification.MulticlassAccuracy,
                                   num_classes=self.datamodule.cls_num_classes,
                                   average='micro'))

    @staticmethod
    def _copy_target_tensor_fn(target: torch.Tensor, batch: Dict[str, torch.Tensor]):
        target[:batch['label'].shape[0]].copy_(batch['label'])

    @staticmethod
    def _create_target_tensor_fn(batch_size: int, image_size: Tuple[int, int]):
        return torch.zeros(batch_size, dtype=torch.long)

    def _forward_task(self, images: torch.Tensor, targets: torch.Tensor, latent: torch.Tensor, is_done: bool,
                      with_loss_and_grad: bool, mode: str, distilled_target: Optional[torch.Tensor] = None):
        out = self.mae.forward_head(latent)

        if is_done:
            self.log_metric(mode, 'accuracy', out.detach(), targets, on_epoch=True, batch_size=latent.shape[0])

        if distilled_target is not None:
            if is_done:
                self.log_metric('train', 'teacher_accuracy', distilled_target.argmax(dim=-1), targets,
                                on_epoch=True,
                                batch_size=latent.shape[0])
            log_target = torch.nn.functional.log_softmax(distilled_target, dim=-1)
            log_out = torch.nn.functional.log_softmax(out, dim=-1)
            kl = torch.nn.functional.kl_div(input=log_out, target=log_target, log_target=True,
                                            reduction='none').sum(dim=-1)
            score = -kl
            loss = kl.mean()
        else:
            if mode == 'train':
                score_target = targets
            else:
                # use model output for early stopping in validation
                score_target = out.argmax(dim=-1)
            score = torch.nn.functional.softmax(out, dim=-1)
            score = score[torch.arange(score.shape[0]), score_target]
            score = score
            loss = self.loss_fn(out, targets)

        return out, loss, score.reshape(score.shape[0], 1)


class SegmentationRlMAE(BaseRlMAE):
    checkpoint_metric = 'val/mPA'
    checkpoint_metric_mode = 'max'
    decoder_type = 'segment'

    def __init__(self, datamodule: BaseDataModule, *args, **kwargs) -> None:
        assert isinstance(datamodule, BaseSegmentationDataModule)
        self.num_classes = datamodule.num_classes
        self.decoder_out_channels = self.num_classes
        self.ignore_label = datamodule.ignore_label

        super().__init__(datamodule, *args, **kwargs)

        self.define_metric('mPA', partial(torchmetrics.classification.MulticlassAccuracy,
                                          num_classes=self.num_classes,
                                          ignore_index=self.ignore_label,
                                          average='macro',
                                          multidim_average='global'))
        self.define_metric('PA', partial(torchmetrics.classification.MulticlassAccuracy,
                                         num_classes=self.num_classes,
                                         ignore_index=self.ignore_label,
                                         average='micro',
                                         multidim_average='global'))
        self.define_metric('mIoU', partial(torchmetrics.classification.MulticlassJaccardIndex,
                                           num_classes=self.num_classes,
                                           ignore_index=self.ignore_label,
                                           average='macro'))
        self.define_metric('teacher_mPA', partial(torchmetrics.classification.MulticlassAccuracy,
                                                  num_classes=self.num_classes,
                                                  ignore_index=self.ignore_label,
                                                  average='macro',
                                                  multidim_average='global'))

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_label)

    @staticmethod
    def _copy_target_tensor_fn(target: torch.Tensor, batch: Dict[str, torch.Tensor]):
        target[:batch['mask'].shape[0]].copy_(batch['mask'])

    @staticmethod
    def _create_target_tensor_fn(batch_size: int, image_size: Tuple[int, int]):
        return torch.zeros((batch_size, image_size[0], image_size[1]), dtype=torch.long)

    def _forward_task(self, images: torch.Tensor, targets: torch.Tensor, latent: torch.Tensor, is_done: bool,
                      with_loss_and_grad: bool, mode: str, distilled_target: Optional[torch.Tensor] = None):
        out = self.mae.forward_decoder(latent)
        # out = self.mae.unpatchify(out)
        pred = out.argmax(dim=1).detach()

        if is_done:
            self.log_metric(mode, 'mPA', pred, targets, on_epoch=True, batch_size=latent.shape[0])
            self.log_metric(mode, 'PA', pred, targets, on_epoch=True, batch_size=latent.shape[0])
            self.log_metric(mode, 'mIoU', pred, targets, on_epoch=True, batch_size=latent.shape[0])

        if distilled_target is not None:
            if is_done:
                self.log_metric('train', 'teacher_mPA', distilled_target.argmax(dim=1), targets,
                                on_epoch=True,
                                batch_size=distilled_target.shape[0])
            log_target = torch.nn.functional.log_softmax(distilled_target, dim=1)
            log_out = torch.nn.functional.log_softmax(out, dim=1)
            kl = torch.nn.functional.kl_div(input=log_out, target=log_target, log_target=True,
                                            reduction='none').mean(dim=-1).mean(dim=-1).sum(dim=-1)
            score = -kl
            loss = kl.mean()
        else:
            loss = self.loss_fn(out, targets).mean(dim=-1).mean(dim=-1)
            score = -loss
            loss = loss.mean()

        return out, loss, score.reshape(score.shape[0], 1)

    def teacher_training_step(self, batch, mode):
        pred = self.teacher_model(batch['image'])['out']
        self.log_metric(mode, 'mPA', pred.detach(), batch['mask'], on_epoch=True, batch_size=pred.shape[0])
        self.log_metric(mode, 'PA', pred.detach(), batch['mask'], on_epoch=True, batch_size=pred.shape[0])
        self.log_metric(mode, 'mIoU', pred.detach(), batch['mask'], on_epoch=True, batch_size=pred.shape[0])
        if mode == 'train':
            loss = self.loss_fn(pred, batch['mask']).mean()
            self.log(name='train/backbone_loss', value=loss.item(), on_step=True, on_epoch=False)
            _, _, _, optimizer_backbone = self.optimizers()
            optimizer_backbone.zero_grad()
            self.manual_backward(loss)
            optimizer_backbone.step()
            _, _, _, scheduler_backbone = self.lr_schedulers()
            scheduler_backbone.step()
