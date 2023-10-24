from functools import partial
from typing import Any, Dict

import torch
import torchmetrics
from torch import nn

from architectures.base import BaseArchitecture
from architectures.glimpse_mae import BaseGlimpseMae
from architectures.glimpse_selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from datasets.base import BaseDataModule
from datasets.classification import BaseClassificationDataModule


class ClassificationMae(BaseGlimpseMae):

    def __init__(self, args, datamodule):
        super().__init__(args, datamodule)
        assert isinstance(datamodule, BaseClassificationDataModule)
        self.num_classes = datamodule.num_classes

        self.define_metric('accuracy',
                           partial(torchmetrics.classification.MulticlassAccuracy,
                                   num_classes=self.num_classes,
                                   average='micro'))

        self.head = nn.Sequential(
            nn.Linear(1024, self.num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward_one(self, x, mask_indices, mask, glimpses) -> Dict[str, torch.Tensor]:
        out = super().forward_one(x, mask_indices, mask, glimpses)
        latent = out['latent'][:, 0, :]  # get cls token
        out['classification'] = self.head(latent)
        return out

    def calculate_loss_one(self, out, batch):
        rec_loss = self.mae.forward_loss(batch[0], out['out'], out['mask'] if self.masked_loss else None)
        cls_loss = self.criterion(out['classification'], batch[1])
        return rec_loss * 0.01 + cls_loss

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)
        self.log_metric(mode, 'accuracy', out['classification'], batch[1])


class RandomClsMae(ClassificationMae):
    glimpse_selector_class = RandomGlimpseSelector


class CheckerboardClsMae(ClassificationMae):
    glimpse_selector_class = CheckerboardGlimpseSelector


class AttentionClsMae(ClassificationMae):
    glimpse_selector_class = AttentionGlimpseSelector
