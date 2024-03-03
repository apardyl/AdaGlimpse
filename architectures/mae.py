# Modified from the official MAE implementation, original copyright info bellow

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from architectures.mae_utils import get_2dplus_sincos_pos_embed, PatchEmbedElastic, \
    get_2dplus_sincos_pos_embed_coords, Layer_scale_init_Block, get_2d_sincos_pos_embed, VisionTransformerUpHead


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_rate: float = 0.,
                 fc_norm: Optional[bool] = None, global_pool: str = 'token', num_classes: int = 1000,
                 decoder_type='standard'):
        super().__init__()

        assert global_pool in ('', 'avg', 'token')
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.global_pool = global_pool

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbedElastic(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = (
            self.patch_embed.img_size[0] // self.patch_embed.patch_size[0],
            self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Layer_scale_init_Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_type = decoder_type
        if decoder_type != 'none':
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                  requires_grad=False)  # fixed sin-cos embedding

            self.decoder_blocks = nn.ModuleList([
                Layer_scale_init_Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                       norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)

            if decoder_type == 'standard':
                self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans,
                                              bias=True)  # decoder to patch
            elif decoder_type == 'segment':
                self.decoder_segment = VisionTransformerUpHead(out_channel=out_chans, embed_dim=decoder_embed_dim,
                                                               grid_shape=self.grid_size)
            else:
                assert False, 'unsupported decoder type'
            # --------------------------------------------------------------------------

            self.norm_pix_loss = norm_pix_loss

            self.decoder_output_tokens = self.decoder_pos_embed.shape[1] - 1

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()
        self.aux_latent = None

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2dplus_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size)
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        if self.decoder_type != 'none':
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size,
                                                        cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        if self.decoder_type != 'none':
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        c = imgs.shape[1]
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0
        assert imgs.shape[3] % p == 0
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = self.grid_size[0]
        w = self.grid_size[1]
        c = int(x.shape[2] / p ** 2)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_encoder(self, x, patch_indices=None, pad_mask=None, coords=None, aux_latent_layer=None):
        # embed patches
        x = self.patch_embed(x)
        N, L, D = x.shape  # batch, length, dim
        # add pos embed w/o cls token
        if coords is None:
            pos_embed = self.pos_embed
        else:
            pos_embed = get_2dplus_sincos_pos_embed_coords(self.patch_embed.embed_dim, coords, cls_token=False)

        x = x + pos_embed

        if patch_indices is not None:
            x = x.gather(1, patch_indices.unsqueeze(2).repeat(1, 1, x.shape[2])).reshape(N, -1, D)
            # Calculate pad_mask
            sorted_indices, indices = torch.sort(patch_indices, dim=1)
            is_overlap = sorted_indices[:, :-1] == sorted_indices[:, 1:]
            is_overlap = torch.cat((torch.full_like(sorted_indices[:, :1], fill_value=False), is_overlap), dim=1)
            pad_mask = torch.gather(is_overlap, 1, indices)

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for idx, blk in enumerate(self.blocks):
            if aux_latent_layer is not None and idx == aux_latent_layer:
                self.aux_latent = x
            x = blk(x, pad_mask)
        x = self.norm(x)

        return x, pos_embed

    def forward_decoder(self, x, mask=None, patch_indices=None):
        # embed tokens
        x = self.decoder_embed(x)
        known_tokens = x.shape[1]
        if mask is not None:
            # append mask tokens to sequence
            x_ = self.mask_token.repeat(*mask.shape, 1)
            x_.scatter_(1, patch_indices.unsqueeze(2).repeat(1, 1, x_.shape[2]), x[:, 1:, :])
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            # add pos embed
            x = x + self.decoder_pos_embed
        else:
            # treat all tokens as missing
            x_ = self.mask_token.repeat(x.shape[0], self.decoder_pos_embed.shape[1] - 1, 1)
            x_ = x_ + self.decoder_pos_embed[:, 1:, :]  # add pos embed
            x = torch.cat([x, x_], dim=1)

            # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        if mask is not None:
            # remove cls token
            x = x[:, 1:, :]
        else:
            # remove all known tokens
            x = x[:, known_tokens:, :]

        # predictor projection
        if self.decoder_type == 'standard':
            x = self.decoder_pred(x)
        elif self.decoder_type == 'segment':
            x = self.decoder_segment(x)

        return x

    def forward_reconstruction_loss(self, imgs, pred, mask=None, mean=True):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if not mean:
            return loss.sum(dim=-1, keepdim=True)  # loss per item in batch
        if mask is not None:
            mask_neg = ~mask
            return (loss * mask_neg).sum() / mask_neg.sum()  # mean loss on removed patches
        else:
            return loss.sum() / pred.shape[1]  # mean loss on all patches

    def forward_head(self, x, pre_logits: bool = False):
        x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def reconstruct(self, pred, target, mask):
        with torch.no_grad():
            pred_img = pred.detach().clone()
            pred_img[mask, :] = self.patchify(target)[mask, :]
            pred_img = self.unpatchify(pred_img)
            return pred_img

    @torch.no_grad()
    def segmentation_output(self, pred):
        pred = self.unpatchify(pred)
        return torch.argmax(pred, dim=1)

    @property
    def decoder_attn_scores(self):
        return torch.stack([block.attn.attn_scores for block in self.decoder_blocks], dim=0)

    @property
    def encoder_attn_scores(self):
        return torch.stack([block.attn.attn_scores for block in self.blocks], dim=0)

    @torch.no_grad()
    def encoder_attention_rollout(self, head_fusion="mean", discard_ratio=0.5):
        attentions = self.encoder_attn_scores
        if attentions.shape[-1] == 1:
            return torch.ones(attentions.shape[1], 0, 1, dtype=attentions.dtype, device=attentions.device)
        result = torch.eye(attentions.shape[3], device=attentions.device).unsqueeze(0).repeat(
            (attentions.shape[1], 1, 1))

        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.amax(dim=1)
            elif head_fusion == "min":
                attention_heads_fused = attention.amin(dim=1)
            else:
                raise "Attention head fusion type Not supported"

            att_head_shape = attention_heads_fused.shape
            flat = attention_heads_fused.reshape(attention_heads_fused.shape[0], -1)
            _, indices = flat.topk(int(flat.shape[1] * discard_ratio), dim=1, largest=False)
            mask = torch.ones_like(flat)
            mask.scatter_(dim=1, index=indices, value=0)
            mask[:, 0] = 1
            flat = flat * mask
            attention_heads_fused = flat.reshape(att_head_shape)

            skip_link = torch.eye(attention_heads_fused.size(-1), device=attentions.device).unsqueeze(0).repeat(
                (attentions.shape[1], 1, 1))
            a = (attention_heads_fused + 1.0 * skip_link) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)

        mask = result[:, 0, 1:]
        mask = torch.nn.functional.softmax(mask, dim=-1)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask.detach()


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
