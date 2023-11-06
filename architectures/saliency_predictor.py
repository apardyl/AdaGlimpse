import sys

import torch
from torch import nn

from architectures.base import BaseArchitecture
from architectures.mae import mae_vit_base_patch16
from datasets.base import BaseDataModule


class SaliencyPredictor(BaseArchitecture):
    def __init__(self, datamodule: BaseDataModule, teacher_path=None, pretrained_path=None, predictor_path=None,
                 **kwargs):
        super().__init__(datamodule, **kwargs)

        self.teacher = mae_vit_base_patch16(img_size=datamodule.image_size)
        if teacher_path is not None:
            print(self.load_teacher(teacher_path), file=sys.stderr)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.predictor = mae_vit_base_patch16(img_size=datamodule.image_size, num_classes=14 * 14)

        if predictor_path:
            print(self.load_pretrained_elastic(predictor_path), file=sys.stderr)

        if self.compile_model:
            self.predictor = torch.compile(self.predictor, mode='reduce-overhead')
            self.teacher = torch.compile(self.teacher, mode='reduce-overhead')

        if pretrained_path:
            print(self.load_pair(pretrained_path), file=sys.stderr)

        self.criterion = nn.MSELoss(reduction='sum')

    def _all_params(self):
        return self.predictor.parameters()

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(SaliencyPredictor.__name__)
        parser.add_argument('--teacher-path',
                            help='path to pretrained MAE or ViT weights',
                            type=str,
                            default='./deit_3_base_224_1k.pth')
        parser.add_argument('--predictor-path',
                            help='path to pretrained MAE or ViT weights',
                            type=str,
                            default='./elastic-224-30random70grid.pth')
        parser.add_argument('--pretrained-path',
                            help='path to pretrained MAE or ViT weights',
                            type=str,
                            default=None)
        return parent_parser

    def load_teacher(self, path):
        checkpoint = torch.load(path, map_location='cpu')["model"]
        return self.teacher.load_state_dict(checkpoint, strict=False)

    def load_pair(self, path):
        checkpoint = torch.load(path, map_location='cpu')["state_dict"]
        return self.load_state_dict(checkpoint, strict=True)

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

        self.teacher.eval()
        with torch.no_grad():
            teacher_out = self.teacher.forward_head(self.teacher.forward_encoder(image))
            saliency = self.teacher.encoder_attention_rollout(discard_ratio=0.9).detach().reshape(image.shape[0], -1)

        pred = self.predictor.forward_head(self.predictor.forward_encoder(patches, coords=coords))

        loss = self.criterion(pred, saliency)

        return {
            'teacher_out': teacher_out,
            'saliency': saliency,
            'out': pred,
            'loss': loss
        }
