from functools import partial

import torch
# noinspection PyProtectedMember
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn
from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from architectures.mae_utils import Layer_scale_init_Block


class ObservationNet(nn.Module):
    def __init__(self, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768, hidden_dim=256):
        super().__init__()

        self.rl_embed = nn.Linear(embed_dim, hidden_dim)

        self.depth = 2

        self.blocks = nn.ModuleList([
            Layer_scale_init_Block(hidden_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                   norm_layer=norm_layer)
            for _ in range(self.depth)])

        if self.depth == 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU()
            )

    def forward(self, observation, mask):
        observation = self.rl_embed(observation)

        for block in self.blocks:
            observation = block(observation, pad_mask=mask)

        # masked global pooling.
        mask = torch.cat([torch.zeros(mask.shape[0], 1, 1, dtype=mask.dtype, device=mask.device), mask], dim=1)
        mask = 1 - mask
        observation = (observation * mask).sum(dim=1) / mask.sum(dim=1)

        return self.net(observation)


class ActorNet(nn.Module):
    def __init__(self, action_dim, norm_layer, hidden_dim, observation_net, glimpse_net):
        super().__init__()

        self.observation_net = observation_net
        self.glimpse_net = glimpse_net

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
            NormalParamExtractor(scale_mapping="biased_softplus_0.5"),
        )

    def forward(self, observation, mask, coords):
        observation = self.observation_net(observation, mask)
        coords = self.glimpse_net(coords.reshape(coords.shape[0], -1))
        observation = torch.cat([observation, coords], dim=-1)
        return self.head(observation)


class QValueNet(nn.Module):
    def __init__(self, action_dim, norm_layer, hidden_dim, observation_net, glimpse_net):
        super().__init__()

        self.observation_net = observation_net
        self.glimpse_net = glimpse_net

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation, mask, coords, action):
        observation = self.observation_net(observation, mask)
        coords = self.glimpse_net(coords.reshape(coords.shape[0], -1))
        action = self.action_net(action)
        observation = torch.cat([observation, coords, action], dim=-1)
        return self.head(observation)


class TransformerActorCritic(nn.Module):
    def __init__(self, action_dim=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768, hidden_dim=192,
                 patch_num=14 * 4):
        super().__init__()

        self.observation_net = ObservationNet(norm_layer=norm_layer, embed_dim=embed_dim, hidden_dim=hidden_dim)

        self.glimpse_net = nn.Sequential(
            nn.Linear(patch_num * 4, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU()
        )

        self.actor_net = ActorNet(
            action_dim=action_dim, norm_layer=norm_layer, hidden_dim=hidden_dim,
            observation_net=self.observation_net, glimpse_net=self.glimpse_net
        )

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                module=self.actor_net,
                in_keys=["observation", "mask", "coords"],
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

        self.qvalue_net = QValueNet(
            action_dim=action_dim, norm_layer=norm_layer, hidden_dim=hidden_dim,
            observation_net=self.observation_net, glimpse_net=self.glimpse_net
        )

        self.qvalue_module = ValueOperator(
            module=self.qvalue_net,
            in_keys=["observation", "mask", "coords", "action"],
        )
