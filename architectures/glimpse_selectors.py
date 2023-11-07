import abc

import torch
import torch.nn as nn

import torchvision.transforms.functional as TF

from architectures.mae import mae_vit_base_patch16


class BaseGlimpseSelector(abc.ABC):
    def __init__(self, model, images, glimpse_size=2):
        self.glimpse_size = glimpse_size
        assert len(images.shape) == 4 and images.shape[1] == 3
        self._images = images
        self._batch_size = images.shape[0]

        self.grid_h = model.mae.grid_size[0]
        self.grid_w = model.mae.grid_size[1]

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(BaseGlimpseSelector.__name__)
        parser.add_argument('--glimpse-size',
                            help='size of a glimpse (in number of patches)',
                            type=int,
                            default=2)

        return parent_parser


class PseudoElasticGlimpseSelector(BaseGlimpseSelector):
    def __call__(self, selection_mask, coords: torch.Tensor):
        assert self.glimpse_size == 2

        B = selection_mask.shape[0]

        # calculate sampling weights for next glimpse
        next_mask = nn.functional.avg_pool2d(selection_mask, kernel_size=self.glimpse_size, stride=1, padding=0)
        next_mask = nn.functional.pad(next_mask,
                                      (0, self.grid_w - next_mask.shape[3], 0, self.grid_h - next_mask.shape[2]),
                                      mode='constant', value=0)
        assert next_mask.shape == (B, 1, self.grid_h, self.grid_w)

        next_mask = next_mask.reshape(B, -1)
        # select next glimpse
        glimpses = torch.argmax(next_mask, dim=1, keepdim=True).unsqueeze(1)
        # B x 1 x 1
        glimpses_h = glimpses.div(self.grid_h, rounding_mode='floor')
        glimpses_w = glimpses - glimpses_h * self.grid_h

        glimpses_h = torch.cat([
            glimpses_h + i for i in range(self.glimpse_size) for j in range(self.glimpse_size)
        ], dim=1)
        glimpses_w = torch.cat([
            glimpses_w + j for i in range(self.glimpse_size) for j in range(self.glimpse_size)
        ], dim=1)
        glimpses_h = glimpses_h * 16
        glimpses_w = glimpses_w * 16
        glimpses_s = torch.ones_like(glimpses_h) * 16

        glimpses = torch.cat([glimpses_h, glimpses_w, glimpses_s], dim=2)
        return glimpses


class STAMLikeGlimpseSelector(BaseGlimpseSelector):

    def __init__(self, model, images, glimpse_size=2):
        super().__init__(model, images, glimpse_size)
        assert self.glimpse_size == 2

        self.mask = torch.ones((self._batch_size, 7 * 7), device=images.device, dtype=torch.long)

    def __call__(self, selection_mask, coords: torch.Tensor):
        assert self.glimpse_size == 2  # WIP

        selection_mask = TF.resize(selection_mask, [7, 7]).squeeze(1)
        selection_mask = selection_mask.reshape(self._batch_size, -1)

        selection_mask *= self.mask
        # select next glimpse
        glimpses = torch.argmax(selection_mask, dim=1, keepdim=True)
        # B x 1 x 1
        self.mask = self.mask.scatter(dim=-1, index=glimpses, value=0)
        glimpses = glimpses.unsqueeze(1)

        glimpses_h = glimpses.div(7, rounding_mode='floor')
        glimpses_w = glimpses - glimpses_h * 7

        glimpses_h = glimpses_h * 2
        glimpses_w = glimpses_w * 2

        glimpses_h = torch.cat([
            glimpses_h + i for i in range(self.glimpse_size) for j in range(self.glimpse_size)
        ], dim=1)
        glimpses_w = torch.cat([
            glimpses_w + j for i in range(self.glimpse_size) for j in range(self.glimpse_size)
        ], dim=1)
        glimpses_h = glimpses_h * 16
        glimpses_w = glimpses_w * 16
        glimpses_s = torch.ones_like(glimpses_h) * 16

        glimpses = torch.cat([glimpses_h, glimpses_w, glimpses_s], dim=2)
        return glimpses


class DivideFourGlimpseSelector(BaseGlimpseSelector):

    def __init__(self, model, images, glimpse_size=2):
        super().__init__(model, images, glimpse_size)
        assert self.glimpse_size == 2
        # use 16x16 grid for easy division by 2, virtual patch size = 14

        start_size = 4
        samples_batch = [[
            [y, x, start_size]
            for y in range(0, 16, start_size)
            for x in range(0, 16, start_size)
        ] for _ in range(self._batch_size)]
        self.samples_batch = samples_batch

    def __call__(self, selection_mask, coords: torch.Tensor):
        selection_mask = TF.resize(selection_mask, [16, 16]).squeeze(1)
        batch_glimpses = []
        for idx in range(self._batch_size):
            samples = self.samples_batch[idx]
            candidates = [
                (selection_mask[idx, s[0]:s[0] + s[2], s[1]:s[1] + s[2]].sum().item(), s) for s in samples if s[2] > 1
            ]
            y, x, s = list(sorted(candidates, reverse=True))[0][1]
            q = s // 2
            glimpses = [
                [i, j, q] for i, j in [(y, x), (y + q, x), (y, x + q), (y + q, x + q)]
            ]
            samples.extend(glimpses)
            batch_glimpses.append(glimpses)
        return torch.tensor(batch_glimpses, dtype=torch.long, device=self._images.device) * 14  # virtual patch size


class ElasticAttentionMapEntropy:
    def __init__(self, model, attention_layer=7, **_):
        self.attention_layer = attention_layer

        self.model = model

        self.grid_h = model.mae.grid_size[0]
        self.grid_w = model.mae.grid_size[1]

    def __call__(self, *_):
        with torch.no_grad():
            # get self attention weights
            attn = self.model.mae.decoder_attn_scores[self.attention_layer][..., -self.model.mae.decoder_output_tokens:,
                   -self.model.mae.decoder_output_tokens:]
            B = attn.shape[0]

            # set attention weights to known patches to 0
            # attn = attn * (~current_mask).reshape((B, 1, -1, 1))

            # calculate entropy of attention weights for each output patch
            entropy = (-attn * torch.log2(attn))
            del attn
            entropy = torch.nan_to_num(entropy, nan=0)
            entropy = entropy.sum((1, 3))
            entropy = entropy.reshape(shape=(B, 1, self.grid_h, self.grid_w))

            return entropy.detach().clone()


class ElasticSaliencyMap:
    def __init__(self, model, **_):
        self.predictor = mae_vit_base_patch16(img_size=model.mae.patch_embed.img_size, num_classes=14 * 14)
        checkpoint = torch.load('/tmp/epoch=132-step=665665.ckpt')['state_dict']
        checkpoint = {k[len('predictor.'):]: v for k, v in checkpoint.items() if k.startswith('predictor')}
        self.predictor.load_state_dict(checkpoint)
        self.predictor = self.predictor.to(model.device)
        self.predictor.eval()

        self.grid_h = model.mae.grid_size[0]
        self.grid_w = model.mae.grid_size[1]

    def __call__(self, patches, coords):
        self.predictor = self.predictor.to(patches.device)
        with torch.no_grad():
            sal_map = self.predictor.forward_head(self.predictor.forward_encoder(patches.to(torch.float32), coords=coords))
            return sal_map.reshape(shape=(patches.shape[0], 1, self.grid_h, self.grid_w))
