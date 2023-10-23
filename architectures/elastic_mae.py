import sys
from abc import ABC

import torch

from architectures.base import BaseArchitecture
from architectures.mae import mae_vit_base_patch16
from datasets.base import BaseDataModule


class ElasticMae(BaseArchitecture, ABC):
    def __init__(self, datamodule: BaseDataModule, out_chans=3, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.mae = mae_vit_base_patch16(img_size=datamodule.image_size, out_chans=out_chans)

        print(self.load_pretrained_elastic('./elastic-224-30random70grid.pth'), file=sys.stderr)

        self.debug = False

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(ElasticMae.__name__)

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


if __name__ == '__main__':
    from datasets.elastic import ElasticImageNet1kClassification

    datamodule = ElasticImageNet1kClassification(data_dir='/home/adam/datasets/imagenet', num_workers=0)
    datamodule.setup('fit')
    model = BaseElasticMae(datamodule)

    batch = next(iter(datamodule.train_dataloader()))
    out = model.forward(batch)

    print(out)
