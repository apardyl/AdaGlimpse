import torch
from timm.models.vision_transformer import vit_small_patch16_224
from torch import nn

from architectures.base import BaseArchitecture
from architectures.mae import mae_vit_large_patch16
from datasets.base import BaseDataModule


class SaliencyPredictor(BaseArchitecture):
    def __init__(self, datamodule: BaseDataModule, lr=1.5e-4, min_lr=1e-8, warmup_epochs=10, weight_decay=0,
                 epochs=100, load_train_data=True):
        super().__init__(datamodule, lr, min_lr, warmup_epochs, weight_decay, epochs)

        if load_train_data:
            self.teacher = mae_vit_large_patch16()
            self.teacher.load_state_dict(torch.load('mae_finetuned_vit_large.pth', map_location='cpu'))

        self.predictor = vit_small_patch16_224(pretrained=True)
        self.predictor.head = nn.Linear(self.embed_dim, 14 * 14)

    def forward(self, batch, compute_loss=False):
        out = self.predictor(batch[0])
        return {
            'out': self.predictor(batch[0])
        }
