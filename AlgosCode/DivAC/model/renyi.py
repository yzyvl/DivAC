import torch
import torch.nn as nn


def weights_init(n):
    if isinstance(n, nn.Linear):
        nn.init.xavier_uniform_(n.weight, gain=1)
        nn.init.constant_(n.bias, 0)


class RenyiDivModel(nn.Module):
    min_log_var = -5
    max_log_var = -1

    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super(RenyiDivModel, self).__init__()

        self.dim = state_dim + action_dim

        self.l1 = nn.Linear(self.dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 2 * state_dim)

        self.l4 = nn.Linear(state_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 2 * state_dim)

        self.device = device

        self.apply(weights_init)

    # @staticmethod
    def swish(self, x):
        """swish activation: x * sigmoid(x)"""
        x = x.to(self.device)
        return x * torch.sigmoid(x)

    def forward(self, state, action, next_state):
        x = torch.cat([state, action], dim=-1)
        x = self.swish(self.l1(x))
        x = self.swish(self.l2(x))
        x = self.l3(x)
        mu0, log_var0 = torch.split(x, x.size(-1) // 2, dim=-1)
        log_var0 = torch.sigmoid(log_var0)  # in [0, 1]
        log_var0 = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var0
        var0 = torch.exp(log_var0)  # normal scale, not log

        y = self.swish(self.l4(next_state))
        y = self.swish(self.l5(y))
        y = self.l6(y)
        mu1, log_var1 = torch.split(y, y.size(-1) // 2, dim=-1)
        log_var1 = torch.sigmoid(log_var1)  # in [0, 1]
        # log_var1 = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var1
        var1 = torch.exp(log_var1)  # normal scale, not log

        return mu0, var0, mu1, var1

    def loss(self, state, action, next_state, alpha=0.9):
        mu0, var0, mu1, var1 = self.forward(state, action, next_state)

        var_alpha = (1-alpha)*(var0**2) + alpha*(var1**2)   # > 0
        item1 = (alpha*(mu1-mu0)**2) / (2*var_alpha)
        item2 = (var_alpha.sqrt().log()-(var0.sqrt()**(1-alpha)).log()-(var1**alpha).log()) / (1-alpha)
        renyi_loss = torch.mean(item1 + item2)
        # print(renyi_loss)

        return renyi_loss

    def div(self, state, action, next_state, alpha):
        mu0, var0, mu1, var1 = self.forward(state, action, next_state)

        var_alpha = (1 - alpha) * (var0 ** 2) + alpha * (var1 ** 2)  # > 0
        item1 = (alpha * (mu1 - mu0) ** 2) / (2 * var_alpha)
        item2 = (var_alpha.sqrt().log() - (var0.sqrt() ** (1 - alpha)).log() - (var1 ** alpha).log()) / (1 - alpha)
        renyid = torch.mean(item1 + item2, dim=-1, keepdim=True)

        return renyid
