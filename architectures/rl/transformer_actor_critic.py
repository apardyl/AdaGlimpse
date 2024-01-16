from functools import partial

import torch
# noinspection PyProtectedMember
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn
from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal


class ActorNet(nn.Module):
    def __init__(self, action_dim, norm_layer, hidden_dim, patch_num):
        super().__init__()

        self.glimpse_net = nn.Sequential(
            nn.Linear(patch_num * 4, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
        )

        self.glimpse_mask = nn.Parameter(torch.zeros(1, patch_num, 4), requires_grad=True)
        torch.nn.init.normal_(self.glimpse_mask, std=.02)

        self.rollout_net = nn.Sequential(
            nn.Linear(patch_num, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
        )

        self.rollout_mask = nn.Parameter(torch.zeros(1, patch_num, 1), requires_grad=True)
        torch.nn.init.normal_(self.rollout_mask, std=.02)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * action_dim),
            NormalParamExtractor(scale_mapping="biased_softplus_0.5"),
        )

    def forward(self, attention, coords, mask):
        attention = attention + (mask * self.rollout_mask)
        attention = self.rollout_net(attention.reshape(coords.shape[0], -1))
        coords = coords + (mask * self.glimpse_mask)
        coords = self.glimpse_net(coords.reshape(coords.shape[0], -1))
        observation = torch.cat([attention, coords], dim=-1)
        observation = self.head(observation)
        return observation


class QValueNet(nn.Module):
    def __init__(self, action_dim, norm_layer, hidden_dim, patch_num):
        super().__init__()

        self.glimpse_net = nn.Sequential(
            nn.Linear(patch_num * 4, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
        )

        self.glimpse_mask = nn.Parameter(torch.zeros(1, patch_num, 4), requires_grad=True)
        torch.nn.init.normal_(self.glimpse_mask, std=.02)

        self.rollout_net = nn.Sequential(
            nn.Linear(patch_num, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
        )

        self.rollout_mask = nn.Parameter(torch.zeros(1, patch_num, 1), requires_grad=True)
        torch.nn.init.normal_(self.rollout_mask, std=.02)

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh()
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # norm_layer(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, attention, coords, action, mask):
        attention = attention + (mask * self.rollout_mask)
        attention = self.rollout_net(attention.reshape(coords.shape[0], -1))
        coords = coords + (mask * self.glimpse_mask)
        coords = self.glimpse_net(coords.reshape(coords.shape[0], -1))
        action = self.action_net(action)
        observation = torch.cat([attention, coords, action], dim=-1)
        return self.head(observation)


class TransformerActorCritic(nn.Module):
    def __init__(self, action_dim=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768, hidden_dim=128,
                 patch_num=14 * 4):
        super().__init__()

        self.actor_net = ActorNet(
            action_dim=action_dim, norm_layer=norm_layer, hidden_dim=hidden_dim, patch_num=patch_num
        )

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                module=self.actor_net,
                in_keys=["attention", "coords", "mask"],
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
            action_dim=action_dim, norm_layer=norm_layer, hidden_dim=hidden_dim, patch_num=patch_num
        )

        self.qvalue_module = ValueOperator(
            module=self.qvalue_net,
            in_keys=["attention", "coords", "action", "mask"],
        )

        self.actor_net.apply(self._init_weights)
        self.qvalue_net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
