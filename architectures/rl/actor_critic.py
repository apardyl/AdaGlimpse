from functools import partial
from typing import List

import torch
# noinspection PyProtectedMember
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn
from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, norm_layer, num_heads=8):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.GELU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.GELU()
        )

    def forward(self, latent: torch.Tensor, mask: torch.Tensor):
        B, P = latent.shape[0], latent.shape[1]
        latent = self.input_layer(latent)  # B x P x D
        att = self.attention(latent)  # B x P x H
        att = att.transpose(1, 2).reshape(B, self.num_heads, 1, P)  # B x H x 1 x P
        att += mask.reshape(att.shape[0], 1, 1, -1) * -1e8
        att = torch.nn.functional.softmax(att, dim=-1)
        latent = (latent.reshape(B, P, self.num_heads, self.hidden_dim // self.num_heads)
                  .permute(0, 2, 1, 3))  # B x H x P x (D / H)
        latent = torch.matmul(att, latent)  # B x H x 1 x (D / H)
        latent = latent.reshape(B, self.hidden_dim)  # B x H
        return self.output_layer(latent)


class BaseAgentNet(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int, norm_layer: nn.Module, patch_num: int,
                 exclude_inputs=None):
        super().__init__()

        self.exclude_inputs = exclude_inputs
        if self.exclude_inputs is not None:
            input_dims = [x for idx, x in enumerate(input_dims) if idx not in self.exclude_inputs]
        self.input_dims = input_dims
        common_dim = 2 * hidden_dim // len(input_dims)
        self.patch_num = patch_num

        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, common_dim),
                norm_layer(common_dim),
                nn.GELU()
            )
            for dim in input_dims
        ])

        self.pooling = AttentionPooling(input_dim=common_dim * len(input_dims), hidden_dim=hidden_dim,
                                        norm_layer=norm_layer)

    def _base_forward(self, mask: torch.Tensor, *x: torch.Tensor):
        x = list(x)
        if self.exclude_inputs is not None:
            x = [a for idx, a in enumerate(x) if idx not in self.exclude_inputs]

        assert len(x) == len(self.input_layers)

        B = x[0].shape[0]

        for idx in range(len(x)):
            c = x[idx].reshape(B, -1, self.input_dims[idx])
            c = c[:, -self.patch_num:, :]  # remove cls token
            c = self.input_layers[idx](c)
            x[idx] = c

        x = torch.cat(x, dim=-1)
        x = self.pooling(x, mask)
        return x


class ActorNet(BaseAgentNet):
    def __init__(self, action_dim, norm_layer, hidden_dim, patch_num, embed_dim, exclude_inputs=None):
        super().__init__(input_dims=[32, 1, 4, embed_dim], hidden_dim=hidden_dim, norm_layer=norm_layer,
                         patch_num=patch_num, exclude_inputs=exclude_inputs)

        self.patch_net_conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * action_dim),
            NormalParamExtractor(scale_mapping="biased_softplus_0.5"),
        )

    def forward(self, mask, patches, attention, coords, observation):
        patches = (self.patch_net_conv(patches.reshape(-1, 3, 16, 16))
                   .reshape(attention.shape[0], attention.shape[1], -1))
        observation = super()._base_forward(mask, patches, attention, coords, observation)
        observation = self.head(observation)
        return observation


class QValueNet(BaseAgentNet):
    def __init__(self, action_dim, norm_layer, hidden_dim, patch_num, embed_dim, exclude_inputs=None):
        super().__init__(input_dims=[32, 1, 4, embed_dim], hidden_dim=hidden_dim, norm_layer=norm_layer,
                         patch_num=patch_num, exclude_inputs=exclude_inputs)

        self.patch_net_conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, action, mask, patches, attention, coords, observation):
        patches = (self.patch_net_conv(patches.reshape(-1, 3, 16, 16))
                   .reshape(attention.shape[0], attention.shape[1], -1))
        observation = super()._base_forward(mask, patches, attention, coords, observation)
        action = self.action_net(action)
        observation = self.head(torch.cat([observation, action], dim=-1))
        return observation


class ActorCritic(nn.Module):
    def __init__(self, action_dim=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768, hidden_dim=256,
                 patch_num=14 * 4, exclude_inputs=None):
        super().__init__()

        if exclude_inputs is not None:
            exclude_inputs = [
                {
                    "patches": 0,
                    "attention": 1,
                    "coords": 2,
                    "observation": 3
                }[e] for e in exclude_inputs
            ]

        self.actor_net = ActorNet(
            action_dim=action_dim, norm_layer=norm_layer, hidden_dim=hidden_dim, patch_num=patch_num,
            embed_dim=embed_dim, exclude_inputs=exclude_inputs
        )

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                module=self.actor_net,
                in_keys=["mask", "patches", "attention", "coords", "observation"],
                out_keys=["loc", "scale"]
            ),
            spec=BoundedTensorSpec(
                minimum=0,
                maximum=1,
                shape=action_dim
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": 0,
                "max": 1,
            },
            return_log_prob=False,
            default_interaction_type=InteractionType.RANDOM,
        )

        self.qvalue_net = QValueNet(
            action_dim=action_dim, norm_layer=norm_layer, hidden_dim=hidden_dim, patch_num=patch_num,
            embed_dim=embed_dim, exclude_inputs=exclude_inputs
        )

        self.qvalue_module = ValueOperator(
            module=self.qvalue_net,
            in_keys=["action", "mask", "patches", "attention", "coords", "observation"],
        )
