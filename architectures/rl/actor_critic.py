import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn
from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_glimpses: int):
        super().__init__()

        class ActorNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.observation_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.step_net = nn.Sequential(
                    nn.Linear(num_glimpses, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.actor_net = nn.Sequential(
                    nn.Linear(3 * hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2 * action_dim),
                    NormalParamExtractor(scale_mapping="biased_softplus_0.5"),
                )

            def forward(self, observation, step):
                observation = self.observation_net(observation)
                step = torch.nn.functional.one_hot(step.clip(0, num_glimpses - 1), num_classes=num_glimpses).to(
                    observation.dtype)
                step = self.step_net(step.squeeze(1))
                return self.actor_net(torch.cat([observation, step], dim=-1))

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                ActorNet(),
                in_keys=["observation", "step"],
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

        class QValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.observation_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.step_net = nn.Sequential(
                    nn.Linear(num_glimpses, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.action_net = nn.Sequential(
                    nn.Linear(action_dim, hidden_dim // 2),
                    nn.ReLU()
                )
                self.value_net = nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, observation, step, action):
                observation = self.observation_net(observation)
                step = torch.nn.functional.one_hot(step.clip(0, num_glimpses - 1), num_classes=num_glimpses).to(
                    observation.dtype)
                step = self.step_net(step.squeeze(1))
                action = self.action_net(action)
                return self.value_net(torch.cat([observation, step, action], dim=-1))

        self.qvalue_module = ValueOperator(
            module=QValueNet(),
            in_keys=["observation", "step", "action"],
        )
