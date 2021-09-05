import torch
import torch.nn as nn


def weights_init(n):
    if isinstance(n, nn.Linear):
        nn.init.xavier_uniform_(n.weight, gain=1)
        nn.init.constant_(n.bias, 0)


class KLDivModel(nn.Module):
    min_log_var = -5
    max_log_var = -1

    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super(KLDivModel, self).__init__()

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

    def loss(self, state, action, next_state, alpha):
        mu0, var0, mu1, var1 = self.forward(state, action, next_state)

        kl_loss = torch.mean(var1.log10() - var0.log10() + (var0**2+(mu0-mu1)**2)/(2*var1**2) - 0.5)

        #print(kl_loss)

        return kl_loss

    def div(self, state, action, next_state, alpha):
        mu0, var0, mu1, var1 = self.forward(state, action, next_state)

        kld = torch.mean(var1.log10() - var0.log10() + (var0**2+(mu0-mu1)**2)/(2*var1**2) - 0.5, dim=-1, keepdim=True)

        return kld
