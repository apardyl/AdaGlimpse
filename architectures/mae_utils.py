from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from timm.layers import to_2tuple, Format
from timm.models.layers import DropPath, Mlp

from datasets.patch_sampler import GridSampler


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2dplus_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    grid = torch.from_numpy(grid)
    grid = torch.stack([grid[1], grid[0], grid[1] + 1, grid[0] + 1], dim=0)
    pos_embed = get_2dplus_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(end=embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2dplus_sincos_pos_embed_coords(embed_dim, patch_coords, cls_token=False):
    """
    patch_coords: [[x1,y1], [x2,y2], ...]
    """
    b, q, t = patch_coords.shape
    assert t == 4

    patch_coords = patch_coords.reshape((b * q, 4)).float() / 16

    grid = patch_coords.permute(1, 0)
    pos_embed = get_2dplus_sincos_pos_embed_from_grid(embed_dim, grid)

    pos_embed = pos_embed.reshape((b, q, embed_dim))

    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([b, 1, embed_dim], dtype=patch_coords.dtype, device=patch_coords.device), pos_embed], dim=1)
    return pos_embed.detach()


def get_2dplus_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h1 = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 4, grid[0])
    emb_w1 = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 4, grid[1])
    emb_h2 = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 4, grid[2])
    emb_w2 = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 4, grid[3])

    emb = torch.cat([emb_h1, emb_w1, emb_h2, emb_w2], dim=1)  # (H*W, D)
    return emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_scores = None

    def forward(self, x, pad_mask=None):
        """ pad_mask: 0 = keep, 1 = ignore in attention """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if pad_mask is not None:
            attn[..., :, 1:] += pad_mask.reshape(B, 1, 1, -1) * -1e8
        attn = attn.softmax(dim=-1)

        self.attn_scores = attn.detach().clone()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, pad_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), pad_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, pad_mask=None):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), pad_mask))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbedElastic(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, x):
        if len(x.shape) == 4:
            # image input
            B, C, H, W = x.shape
            P = None
            assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
            assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        elif len(x.shape) == 5:
            # patch input
            B, P, C, H, W = x.shape
            assert H == self.patch_size[0]
            assert W == self.patch_size[1]
            x = x.reshape(B * P, C, H, W)
        else:
            assert False

        x = self.proj(x)

        if P is None:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.reshape(B, P, self.embed_dim)
        x = self.norm(x)
        return x


class VisionTransformerUpHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, out_channel, embed_dim=256, grid_shape=(14, 14)):
        super(VisionTransformerUpHead, self).__init__()

        self.grid_shape = grid_shape

        self.conv_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(256),
            nn.GELU(),
            nn.Conv2d(256, out_channel, kernel_size=1, stride=1)
        )

    def forward(self, x):
        n, hw, c = x.shape
        h, w = self.grid_shape
        assert hw == h * w
        x = x.transpose(1, 2).reshape(n, c, h, w)

        x = self.conv_head(x)

        return x
