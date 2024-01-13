from functools import partial

import torch
# noinspection PyProtectedMember
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn
from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from architectures.mae_utils import Layer_scale_init_Block


class ActorNet(nn.Module):
    def __init__(self, action_dim, norm_layer, hidden_dim, rollout_net, glimpse_net):
        super().__init__()

        self.rollout_net = rollout_net
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

    def forward(self, attention, coords):
        attention = self.rollout_net(attention.reshape(coords.shape[0], -1))
        coords = self.glimpse_net(coords.reshape(coords.shape[0], -1))
        observation = torch.cat([attention, coords], dim=-1)
        observation = self.head(observation)
        return observation


class QValueNet(nn.Module):
    def __init__(self, action_dim, norm_layer, hidden_dim, rollout_net, glimpse_net):
        super().__init__()

        self.rollout_net = rollout_net
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

    def forward(self, attention, coords, action):
        attention = self.rollout_net(attention.reshape(coords.shape[0], -1))
        coords = self.glimpse_net(coords.reshape(coords.shape[0], -1))
        action = self.action_net(action)
        observation = torch.cat([attention, coords, action], dim=-1)
        return self.head(observation)


class TransformerActorCritic(nn.Module):
    def __init__(self, action_dim=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768, hidden_dim=256,
                 patch_num=14 * 4):
        super().__init__()

        self.glimpse_net = nn.Sequential(
            nn.Linear(patch_num * 4, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU()
        )

        self.rollout_net = nn.Sequential(
            nn.Linear(patch_num, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU()
        )

        self.actor_net = ActorNet(
            action_dim=action_dim, norm_layer=norm_layer, hidden_dim=hidden_dim,
            rollout_net=self.rollout_net, glimpse_net=self.glimpse_net
        )

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                module=self.actor_net,
                in_keys=["attention", "coords"],
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
            action_dim=action_dim, norm_layer=norm_layer, hidden_dim=hidden_dim,
            rollout_net=self.rollout_net, glimpse_net=self.glimpse_net
        )

        self.qvalue_module = ValueOperator(
            module=self.qvalue_net,
            in_keys=["attention", "coords", "action"],
        )
