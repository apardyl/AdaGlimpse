import sys
from abc import ABC
from functools import partial

import torch
import torchmetrics

from architectures.base import BaseArchitecture
from architectures.glimpse_selectors import PseudoElasticGlimpseSelector, ElasticAttentionMapEntropy, \
    DivideFourGlimpseSelector, STAMLikeGlimpseSelector, ElasticSaliencyMap
from architectures.mae import mae_vit_base_patch16
from architectures.utils import MetricMixin
from datasets.base import BaseDataModule
from datasets.patch_sampler import InteractiveSampler
from datasets.utils import IMAGENET_MEAN, IMAGENET_STD


class ElasticMae(BaseArchitecture, ABC):
    def __init__(self, datamodule: BaseDataModule, out_chans=3, pretrained_path=None, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.mae = mae_vit_base_patch16(img_size=datamodule.image_size, out_chans=out_chans)

        if self.compile_model:
            self.mae = torch.compile(self.mae, mode='reduce-overhead')

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

    def load_pretrained_elastic(self, path="./elastic-224-30random70grid.pth"):
        checkpoint = torch.load(path, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint = checkpoint["model"]
            prefix = ''
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
            prefix = 'mae.'
        else:
            raise NotImplemented()

        if prefix + 'pos_embed' in checkpoint:
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


class _GlimpseElasticMae(ElasticMae):
    def __init__(self, datamodule: BaseDataModule, num_glimpses=12, **kwargs):
        super().__init__(datamodule, **kwargs)
        self.num_glimpses = num_glimpses

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(AMEGlimpseElasticMae.__name__)
        parser.add_argument('--num-glimpses',
                            help='number of glimpses to take',
                            type=int,
                            default=8)
        return parent_parser


class _GlimpseElasticMaeReconstruction(_GlimpseElasticMae, MetricMixin):
    def __init__(self, datamodule: BaseDataModule, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.define_metric('rmse', partial(torchmetrics.MeanSquaredError, squared=False))
        self.register_buffer('imagenet_mean', torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1))

    def __rev_normalize(self, img):
        return torch.clip((img * self.imagenet_std + self.imagenet_mean) * 255, 0, 255)

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)

        pred = self.mae.unpatchify(out['out'])
        pred = self.__rev_normalize(pred)
        target = self.__rev_normalize(batch['image'])

        self.log_metric(mode, 'rmse', pred, target)


class AMEGlimpseElasticMae(_GlimpseElasticMaeReconstruction):
    selection_map_extractor_class = ElasticAttentionMapEntropy

    def __init__(self, datamodule: BaseDataModule, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.patch_sampler_class = InteractiveSampler
        self.extractor = self.selection_map_extractor_class(self)

    def forward(self, batch, compute_loss=True):
        image = batch['image']
        sampler = self.patch_sampler_class(image)
        selector = self.glimpse_selector_class(self, image)
        out = None
        loss = 0

        for step in range(self.num_glimpses - 4):
            latent = self.mae.forward_encoder(sampler.patches, coords=sampler.coords)
            out = self.mae.forward_decoder(latent)
            loss = self.mae.forward_reconstruction_loss(image, out)

            selection_mask = self.extractor(sampler.patches, sampler.coords)
            next_glimpse = selector(selection_mask, sampler.coords)
            sampler.sample(next_glimpse)

        return {'out': out, 'loss': loss, 'coords': sampler.coords}


class SimpleAMEGlimpseElasticMae(AMEGlimpseElasticMae):
    glimpse_selector_class = PseudoElasticGlimpseSelector


class DivideFourGlimpseElasticMae(AMEGlimpseElasticMae):
    glimpse_selector_class = DivideFourGlimpseSelector


class StamlikeGlimpseElasticMae(AMEGlimpseElasticMae):
    glimpse_selector_class = STAMLikeGlimpseSelector


class SaliencyGlimpseElasticMae(_GlimpseElasticMaeReconstruction):
    selection_map_extractor_class = ElasticSaliencyMap
    glimpse_selector_class = None

    def __init__(self, datamodule: BaseDataModule, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.patch_sampler_class = InteractiveSampler
        self.extractor = self.selection_map_extractor_class(self)

    def forward(self, batch, compute_loss=True):
        image = batch['image']
        sampler = self.patch_sampler_class(image)
        selector = self.glimpse_selector_class(self, image)

        for step in range(self.num_glimpses - 4):
            selection_mask = self.extractor(sampler.patches, sampler.coords)
            next_glimpse = selector(selection_mask, sampler.coords)
            sampler.sample(next_glimpse)

        latent = self.mae.forward_encoder(sampler.patches, coords=sampler.coords)
        out = self.mae.forward_decoder(latent)
        loss = self.mae.forward_reconstruction_loss(image, out)

        return {'out': out, 'loss': loss, 'coords': sampler.coords}


class StamlikeSaliencyGlimpseElasticMae(SaliencyGlimpseElasticMae):
    glimpse_selector_class = STAMLikeGlimpseSelector


class DivideFourSaliencyGlimpseElasticMae(SaliencyGlimpseElasticMae):
    glimpse_selector_class = DivideFourGlimpseSelector
