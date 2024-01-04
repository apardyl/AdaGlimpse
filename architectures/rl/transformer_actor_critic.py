from functools import partial

import torch
# noinspection PyProtectedMember
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn
from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from architectures.mae_utils import Layer_scale_init_Block


class ActorNet(nn.Module):
    def __init__(self, action_dim, norm_layer, hidden_dim, observation_net, glimpse_net):
        super().__init__()

        self.observation_net = observation_net
        self.glimpse_net = glimpse_net

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
            NormalParamExtractor(scale_mapping="biased_softplus_0.5"),
        )

    def forward(self, observation, mask, coords):
        observation = self.observation_net(observation.mean(dim=1))
        coords = self.glimpse_net(coords.reshape(coords.shape[0], -1))
        return self.head(torch.cat([observation, coords], dim=-1))


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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation, mask, coords, action):
        observation = self.observation_net(observation.mean(dim=1))
        coords = self.glimpse_net(coords.reshape(coords.shape[0], -1))
        action = self.action_net(action)
        return self.head(torch.cat([observation, coords, action], dim=-1))


class TransformerActorCritic(nn.Module):
    def __init__(self, action_dim=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768, hidden_dim=256,
                 patch_num=14 * 4):
        super().__init__()

        self.observation_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU()
        )

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
