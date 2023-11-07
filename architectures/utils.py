import math
from typing import Dict, Callable, Collection

import torch
import torch.nn as nn


class MaeScheduler(nn.Module):
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.

    # This source code is licensed under the license found in the
    # LICENSE file in the root directory of this source tree.

    def __init__(self, optimizer, lr, warmup_epochs, min_lr, epochs):
        super().__init__()

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.epochs = epochs
        self.optimizer = optimizer

    def step(self, epoch, metrics=None):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * (epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs + 1 - self.warmup_epochs)))

        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr


def dict_to_cpu(tensor_dict: Dict[str, torch.Tensor]):
    return {k: v.detach().clone().cpu() for k, v in tensor_dict.items()}


class MetricMixin:
    def define_metric(self, name: str, metric_constructor: Callable):
        for mode in ['train', 'val', 'test']:
            setattr(self, f'{mode}_{name}', metric_constructor())

    def get_metric(self, mode: str, name: str):
        return getattr(self, f'{mode}_{name}')

    def log_metric(self, mode: str, name: str, *args, on_step: bool = False, on_epoch: bool = True,
                   sync_dist: bool = True, prog_bar: bool = False, **kwargs) -> None:
        self.log(name=f'{mode}/{name}', value=self.get_metric(mode, name)(*args, **kwargs), on_step=on_step,
                 on_epoch=on_epoch, sync_dist=sync_dist, prog_bar=prog_bar)
