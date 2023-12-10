from functools import partial

import torch
# noinspection PyProtectedMember
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn
from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from architectures.mae_utils import Layer_scale_init_Block


class ActorNet(nn.Module):
    def __init__(self, action_dim, norm_layer, embed_dim, hidden_dim,
                 num_heads, mlp_ratio, depth):
        super().__init__()

        self.rl_embed = nn.Linear(embed_dim, hidden_dim, bias=True)

        self.blocks = nn.ModuleList([
            Layer_scale_init_Block(hidden_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                   norm_layer=norm_layer)
            for _ in range(depth)])

        self.head = nn.Sequential(
            norm_layer(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
            NormalParamExtractor(scale_mapping="biased_softplus_0.5"),
        )

    def forward(self, observation, mask):
        observation = self.rl_embed(observation)

        for block in self.blocks:
            observation = block(observation, pad_mask=mask)

        observation = observation[:, 0]
        observation = self.head(observation)
        return observation


class QValueNet(nn.Module):
    def __init__(self, action_dim, norm_layer, embed_dim, hidden_dim,
                 num_heads, mlp_ratio, depth):
        super().__init__()

        self.rl_embed = nn.Linear(embed_dim, hidden_dim, bias=True)

        self.blocks = nn.ModuleList([
            Layer_scale_init_Block(hidden_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                   norm_layer=norm_layer)
            for _ in range(depth)])

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.value_net = nn.Sequential(
            norm_layer(3 * hidden_dim // 2),
            nn.Linear(3 * hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation, mask, action):
        observation = self.rl_embed(observation)

        for block in self.blocks:
            observation = block(observation, pad_mask=mask)
        observation = observation[:, 0]
        action = self.action_net(action)
        return self.value_net(torch.cat([observation, action], dim=-1))


class TransformerActorCritic(nn.Module):
    def __init__(self, action_dim=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768, hidden_dim=192,
                 num_heads=12, mlp_ratio=4, depth=2):
        super().__init__()

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                ActorNet(
                    action_dim=action_dim, norm_layer=norm_layer, embed_dim=embed_dim, hidden_dim=hidden_dim,
                    num_heads=num_heads, mlp_ratio=mlp_ratio, depth=depth
                ),
                in_keys=["observation", "mask"],
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
            default_interaction_type=InteractionType.RANDOM
        )

        self.qvalue_module = ValueOperator(
            module=QValueNet(
                action_dim=action_dim, norm_layer=norm_layer, embed_dim=embed_dim, hidden_dim=hidden_dim,
                num_heads=num_heads, mlp_ratio=mlp_ratio, depth=depth
            ),
            in_keys=["observation", "mask", "action"],
        )
