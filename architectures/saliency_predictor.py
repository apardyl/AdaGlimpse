import sys

import torch

from architectures.base import BaseArchitecture
from architectures.mae import mae_vit_base_patch16
from datasets import ElasticImageNet1k
from datasets.base import BaseDataModule


class SaliencyPredictor(BaseArchitecture):
    def __init__(self, datamodule: BaseDataModule, teacher_path=None, pretrained_path=None, **kwargs):
        super().__init__(datamodule, **kwargs)

        self.teacher = mae_vit_base_patch16(img_size=datamodule.image_size)
        print(self.load_teacher(teacher_path), file=sys.stderr)

        self.predictor = mae_vit_base_patch16(img_size=datamodule.image_size, num_classes=14 * 14)

        if pretrained_path:
            print(self.load_pretrained_elastic(pretrained_path), file=sys.stderr)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(SaliencyPredictor.__name__)
        parser.add_argument('--teacher-path',
                            help='path to pretrained MAE or ViT weights',
                            type=str,
                            default='./deit_3_base_224_1k.pth')
        parser.add_argument('--pretrained-path',
                            help='path to pretrained MAE or ViT weights',
                            type=str,
                            default='./elastic-224-30random70grid.pth')
        return parent_parser

    def load_teacher(self, path):
        checkpoint = torch.load(path, map_location='cpu')["model"]
        return self.teacher.load_state_dict(checkpoint, strict=False)

    def load_pretrained_elastic(self, path=""):
        checkpoint = torch.load(path, map_location='cpu')["model"]

        del checkpoint['pos_embed']
        del checkpoint['head.weight']
        del checkpoint['head.bias']

        return self.predictor.load_state_dict(checkpoint, strict=False)

    def forward(self, batch):
        image = batch['image']
        patches = batch['patches']
        coords = batch['coords']

        teacher_out = self.teacher.forward_head(self.teacher.forward_encoder(image))
        saliency = self.teacher.encoder_attention_rollout().detach()

        pred = self.predictor.forward_head(self.predictor.forward_encoder(patches, coords=coords))

        loss = ((pred - saliency.reshape(pred.shape)) ** 2).mean()

        return {
            'teacher_out': teacher_out,
            'saliency': saliency,
            'out': pred,
            'loss': loss
        }


if __name__ == '__main__':
    data = ElasticImageNet1k(data_dir='/home/adam/datasets/imagenet', train_batch_size=2, eval_batch_size=2)
    data.setup('fit')
    model = SaliencyPredictor(datamodule=data, teacher_path='../deit_3_base_224_1k.pth',
                              pretrained_path='../elastic-224-30random70grid.pth')
    loader = data.val_dataloader()
    batch = next(iter(loader))

    out = model(batch)

    print(out)
