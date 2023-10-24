import abc
import sys
from abc import ABC
from typing import Dict

import torch

from architectures.base import BaseArchitecture
from architectures.mae import mae_vit_base_patch16
from datasets.base import BaseDataModule


class ElasticMae(BaseArchitecture, ABC):
    def __init__(self, datamodule: BaseDataModule, out_chans=3, pretrained_path=None, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.mae = mae_vit_base_patch16(img_size=datamodule.image_size, out_chans=out_chans)

        if pretrained_path:
            print(self.load_pretrained_elastic(pretrained_path), file=sys.stderr)

        self.debug = False

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(ElasticMae.__name__)
        parser.add_argument('--pretrained-path',
                            help='path to pretrained MAE or ViT weights',
                            type=str,
                            default='./elastic-224-30random70grid.pth')
        return parent_parser

    def load_pretrained_elastic(self, path="architectures/mae_vit_l_128x256.pth"):
        checkpoint = torch.load(path, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint = checkpoint["model"]
            prefix = ''
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
            prefix = 'mae.'
        else:
            raise NotImplemented()

        del checkpoint[prefix + 'pos_embed']

        if prefix == '':
            return self.mae.load_state_dict(checkpoint, strict=False)
        else:
            return self.load_state_dict(checkpoint, strict=False)

    def forward(self, batch):
        image = batch['image']
        patches = batch['patches']
        coords = batch['coords']

        latent = self.mae.forward_encoder(patches, coords=coords)
        out = self.mae.forward_decoder(latent)
        loss = self.mae.forward_reconstruction_loss(image, out)

        return {"out": out, "loss": loss}


class GlimpseElasticMae(ElasticMae):
    glimpse_selector_class = None

    def __init__(self, datamodule: BaseDataModule, num_glimpses=8, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.num_glimpses = num_glimpses

        # disable patch sampling in dataset
        datamodule.patch_sampler = None

        assert self.glimpse_selector_class is not None
        self.glimpse_selector = self.glimpse_selector_class(self, **kwargs)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(GlimpseElasticMae.__name__)
        parser.add_argument('--num-glimpses',
                            help='number of glimpses to take',
                            type=int,
                            default=8)
        parent_parser = cls.glimpse_selector_class.add_argparse_args(parent_parser)
        return parent_parser

    @abc.abstractmethod
    def calculate_loss_one(self, out, batch):
        raise NotImplemented()

    def calculate_loss(self, losses, batch):
        return torch.mean(torch.stack(losses))

    def forward_one(self, x, coords) -> Dict[str, torch.Tensor]:
        latent = self.mae.forward_encoder(x, coords=coords)
        out = self.mae.forward_decoder(latent)
        return {
            'out': out,
            'latent': latent,
            'coords': coords
        }

    def forward(self, batch, compute_loss=True):
        image = batch['image']

        coords = []
        patches = []

        loss = 0
        losses = []
        steps = []

        if not self.single_step:
            # zero step (initialize decoder attention weights)
            out = self.forward_one(x, mask_indices, mask, glimpses)
            if self.debug:
                steps.append(dict_to_cpu(out))
        for i in range(self.num_glimpses):
            mask, mask_indices, glimpse = self.glimpse_selector(mask, mask_indices, i)
            glimpses.append(glimpse)
            if self.single_step and i + 1 < self.num_glimpses:
                continue
            out = self.forward_one(x, mask_indices, mask, glimpses)
            if compute_loss:
                loss = self.calculate_loss_one(out, batch)
                losses.append(loss)
            if self.debug:
                steps.append(dict_to_cpu(out | self.glimpse_selector.debug_info))

        if compute_loss and self.sum_losses:
            loss = self.calculate_loss(losses, batch)

        return out | {"losses": losses, "loss": loss, "steps": steps}
