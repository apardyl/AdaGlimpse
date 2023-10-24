from abc import ABC
from typing import Any, Optional

import torchmetrics
from lightning import LightningModule
from torch.optim import AdamW

from architectures.utils import MaeScheduler, MetricMixin
from datasets.base import BaseDataModule


# noinspection PyArgumentList
class BaseArchitecture(LightningModule, MetricMixin, ABC):
    def __init__(self, datamodule: BaseDataModule, lr=1.5e-4, min_lr=1e-8, warmup_epochs=10, weight_decay=0,
                 epochs=100, **_):
        super().__init__()

        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.current_lr = lr

        self.save_hyperparameters(ignore=['datamodule'])

        self.define_metric('loss', torchmetrics.MeanMetric)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(BaseArchitecture.__name__)
        parser.add_argument('--lr',
                            help='learning-rate',
                            type=float,
                            default=1e-3)
        parser.add_argument('--warmup-epochs',
                            help='epochs to warmup LR',
                            type=int,
                            default=10)
        parser.add_argument('--min-lr',
                            help='lower lr bound for cyclic schedulers that hit 0',
                            type=float,
                            default=1e-8)
        parser.add_argument('--weight-decay',
                            help='weight_decay',
                            type=float,
                            default=0.01)
        parser.add_argument('--epochs',
                            help='number of epochs',
                            type=int,
                            default=400)
        return parent_parser

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.95))
        scheduler = MaeScheduler(optimizer=optimizer,
                                 lr=self.lr,
                                 warmup_epochs=self.warmup_epochs,
                                 min_lr=self.min_lr,
                                 epochs=self.epochs)
        scheduler.step(epoch=0)

        lr_schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        ]

        return [optimizer], lr_schedulers

    def lr_scheduler_step(self, scheduler, metric: Optional[Any]) -> None:
        # noinspection PyNoneFunctionAssignment
        self.current_lr = scheduler.step(epoch=self.current_epoch)

    def do_metrics(self, mode, out, batch):
        pass

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = out['loss']
        self.log_metric('train', 'loss', loss, on_step=True, on_epoch=False, sync_dist=False)
        self.log('train/lr', self.current_lr, on_step=False, on_epoch=True, sync_dist=True)
        self.do_metrics('train', out, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log_metric('val', 'loss', out['loss'])
        self.do_metrics('val', out, batch)

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log_metric('test', 'loss', out['loss'])
        self.do_metrics('test', out, batch)
