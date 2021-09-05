import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def weights_init(n):
    if isinstance(n, nn.Linear):
        nn.init.xavier_uniform_(n.weight, gain=1)
        nn.init.constant_(n.bias, 0)


class QNet(nn.Module):
    def __init__(self, d_state, d_action, hidden_size):
        super().__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(d_state + d_action, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(d_state + d_action, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)

        self.apply(weights_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)

        x1 = F.relu(self.linear1(sa))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(sa))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, max_action):
        super().__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

        self.apply(weights_init)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        return self.max_action * torch.tanh(self.l3(a))

    def sample(self, state):
        a = self.forward(state)

        return a, a


class GaussianPolicy(nn.Module):
    def __init__(self, d_state, d_action, hidden_size, action_space=None):
        super().__init__()

        self.linear1 = nn.Linear(d_state, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, d_action)
        self.log_std_linear = nn.Linear(hidden_size, d_action)

        self.apply(weights_init)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale.to(mean.device) + self.action_bias.to(mean.device)

        return action, torch.tanh(mean)
