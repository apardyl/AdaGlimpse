import torch
from torch import nn

EPS = 0.003


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.action_layers = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU()
        )

        self.out_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.apply(self._init_weights)
        self.out_layers[-1].weight.data.uniform_(-EPS, EPS)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s2 = self.state_layers(state)
        a1 = self.action_layers(action)
        x = torch.cat((s2, a1), dim=1)
        x = self.out_layers(x)
        return x


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)
        self.layers[-2].weight.data.uniform_(-EPS, EPS)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        action = self.layers(state)
        return action
