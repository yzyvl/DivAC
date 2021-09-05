import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.distributions import Normal
from utils.normalizer import Normalizer
from torch.distributions.multivariate_normal import MultivariateNormal


def weights_init(n):
    if isinstance(n, nn.Linear):
        nn.init.xavier_uniform_(n.weight, gain=1)
        nn.init.constant_(n.bias, 0)


class RenyiDivModel(nn.Module):
    def __init__(self, d_state, d_action, hidden_size, device):
        super().__init__()

        self.device = device
        self.norm_dist = MultivariateNormal(torch.zeros(d_state), torch.eye(d_state))

        self.linear1 = nn.Linear(d_state + d_action, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        # d_state + d_action
        self.mean_l = nn.Linear(hidden_size, d_state)
        self.log_var_l = nn.Linear(hidden_size, d_state)

        self.linear4 = nn.Linear(d_state, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, d_state)
        self.log_var = nn.Linear(hidden_size, d_state)

        self.apply(weights_init)

    def swish(self, x):
        x = x.to(self.device)
        return x * torch.sigmoid(x)

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=-1)
        x = self.swish(self.linear1(state_action))
        x = self.swish(self.linear2(x))
        x = self.swish(self.linear3(x))

        mean_, log_var_ = self.mean_l(x), self.log_var_l(x)
        log_var_ = torch.sigmoid(log_var_)
        eps = self.norm_dist.sample((mean_.size(0),)).to(self.device)
        z = mean_ + log_var_.exp().sqrt() * eps

        z_s = self.swish(self.linear4(z))
        z_s = self.swish(self.linear5(z_s))
        z_s = self.swish(self.linear6(z_s))

        delta_mu, delta_var = self.mu(z_s), torch.sigmoid(self.log_var(z_s)).exp()

        return mean_, log_var_, z, delta_mu, delta_var

    @staticmethod
    def log_norm(x, mu, std):
        # return - torch.log(torch.tensor(2 * np.pi).sqrt() * std) - (0.5 * ((x - mu) / std) ** 2)
        return -0.5 * torch.log(2 * np.pi * std ** 2) - (0.5 * (1 / (std ** 2)) * (x - mu) ** 2)

    def loss(self, states, actions, delta_states, alpha):
        mean_, log_var_, z, delta_mu, delta_std = self.forward(states, actions)
        q_z = self.log_norm(z, mean_, F.softplus(log_var_).sqrt())
        prior_z = self.log_norm(z, torch.zeros(mean_.size()).to(self.device), torch.ones(mean_.size()).to(self.device))
        likelihood = self.log_norm(delta_states, delta_mu, delta_std.sqrt())

        ratio = (1 - alpha) * (prior_z + likelihood - q_z)
        const = torch.max(ratio)
        renyi = - ((ratio - const).exp().mean().log() + const) / (1 - alpha)

        return renyi

    def renyi_alpha_div(self, states, actions, delta_states, alpha):
        mean_, log_var_, z, delta_mu, delta_std = self.forward(states, actions)
        q_z = self.log_norm(z, mean_, F.softplus(log_var_).sqrt())
        prior_z = self.log_norm(z, torch.zeros(mean_.size()).to(self.device), torch.ones(mean_.size()).to(self.device))
        likelihood = self.log_norm(delta_states, delta_mu, delta_std.sqrt())

        ratio = (1 - alpha) * (prior_z + likelihood - q_z)
        const = torch.max(ratio)
        renyi = - ((ratio - const).exp().mean().log() + const) / (1 - alpha)

        # print(renyi)

        return renyi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KLDivModel(nn.Module):
    min_log_var = -5
    max_log_var = -1

    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super(KLDivModel, self).__init__()

        self.dim = state_dim + action_dim

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 2 * state_dim)

        self.linear5 = nn.Linear(state_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, 2 * state_dim)

        self.device = device

        self.apply(weights_init)

    # @staticmethod
    def swish(self, x):
        """swish activation: x * sigmoid(x)"""
        x = x.to(self.device)
        return x * torch.sigmoid(x)

    def compressor(self, s_a):
        x = self.swish(self.linear1(s_a))
        x = self.swish(self.linear2(x))
        x = self.swish(self.linear3(x))
        x = self.linear4(x)
        mu, log_var = torch.split(x, x.size(-1) // 2, dim=-1)
        log_var = torch.sigmoid(log_var)

        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = log_var.exp().sqrt()
        eps = std.data.new(std.size()).normal_().to(device)
        z = mu + eps * std

        return z

    def predictor(self, z):
        x = self.swish(self.linear5(z))
        x = self.swish(self.linear6(x))
        x = self.swish(self.linear7(x))
        x = self.linear8(x)
        mean, log_var = torch.split(x, x.size(-1) // 2, dim=-1)
        log_var = torch.sigmoid(log_var)  # in [0, 1]
        # log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var
        var = torch.exp(log_var)  # normal scale, not log

        return mean, var

    def feed_forward(self, state, action):
        s_a = torch.cat((state, action), dim=-1)
        mu, log_var = self.compressor(s_a)
        z = self.reparametrize(mu, log_var)
        mean, var = self.predictor(z)

        return mu, log_var, mean, var

    def loss(self, state, action, state_deltas):
        # normalized_state, normalized_action = self.get_model_input(state, action)
        mu, sigma, mean, var = self.feed_forward(state, action)
        # target = self.get_model_target(state_deltas)

        # negative log likelihood
        neg_log_likelihood = torch.mean((mean - state_deltas) ** 2 / var + var.log())
        # kl divergence
        kl_divergence = 0.5 * torch.sum((mu ** 2) + (sigma ** 2) - torch.log((sigma ** 2) + 1e-6) - 1, dim=-1)
        dim = mu.size(-1)
        kl_divergence = torch.mean(kl_divergence / (mu.view(-1, dim).data.shape[0] * dim))
        # assert 0 <= kl_divergence <= 1, 'kl divergence: %.4f' % kl_divergence

        loss = neg_log_likelihood + kl_divergence
        # print(loss)

        return loss

    def kl_div(self, state, action, state_deltas):
        mu, sigma, mean, var = self.feed_forward(state, action)
        neg_log_likelihood = torch.mean((mean - state_deltas) ** 2 / var + var.log())
        kl_divergence = 0.5 * torch.sum((mu ** 2) + (sigma ** 2) - torch.log((sigma ** 2) + 1e-6) - 1, dim=-1)
        dim = mu.size(-1)
        kl_divergence = torch.mean(kl_divergence / (mu.view(-1, dim).data.shape[0] * dim))

        # print(kl_divergence + neg_log_likelihood)
        # div = torch.clamp(kl_divergence + neg_log_likelihood, 0, 100)
        div = kl_divergence + neg_log_likelihood
        # print(div)

        return div

# class EnsembleLayer(nn.Module):
#     def __init__(self, input_size, output_size, ensemble_size, nonlinearity='tanh'):
#         """
#         linear + activation layer
#         :param input_size: size of the input vector
#         :param output_size: size of the output vector
#         :param ensemble_size: number of the models in the ensemble
#         :param nonlinearity: 'swish', 'tanh' or 'leaky_relu'
#         """
#         super(EnsembleLayer, self).__init__()
#         weights = torch.zeros(ensemble_size, input_size, output_size).float()
#         biases = torch.zeros(ensemble_size, 1, output_size).float()
#         for weight in weights:
#             if nonlinearity == 'leaky_relu':
#                 nn.init.kaiming_normal_(weight)
#             else:
#                 nn.init.xavier_normal_(weight)
#
#         self.weights, self.biases = nn.Parameter(weights, requires_grad=True), nn.Parameter(biases, requires_grad=True)
#
#         if nonlinearity == 'swish':
#             self.nonlinearity = self.swish
#         elif nonlinearity == 'leaky_relu':
#             self.nonlinearity = F.leaky_relu
#         elif nonlinearity == 'tanh':
#             self.nonlinearity = torch.tanh
#         else:
#             self.nonlinearity = lambda x: x
#
#     @staticmethod
#     def swish(x):
#         """swish activation: x * sigmoid(x)"""
#         return x * torch.sigmoid(x)
#
#     def forward(self, x):
#         """nonlinearity(weights * x + biases)"""
#         out = torch.baddbmm(self.biases, x.to(device), self.weights)
#
#         return self.nonlinearity(out)
#
#
# class Model(nn.Module):
#     def __init__(self, dim_state, hidden_size, dim_action, n_layers, ensemble_size, nonlinearity):
#         super().__init__()
#         self.dim_state, self.dim_action, self.ensemble_size, self.normalizer = dim_state, dim_action, ensemble_size, None
#
#         compressor_layers, predictor_layers = [], []
#
#         for i_layer in range(n_layers + 1):
#             if i_layer == 0:
#                 layer = EnsembleLayer(dim_state + dim_action, hidden_size, ensemble_size, nonlinearity)
#             elif 0 < i_layer < n_layers:
#                 layer = EnsembleLayer(hidden_size, hidden_size, ensemble_size, nonlinearity)
#             else:
#                 layer = EnsembleLayer(hidden_size, 2 * dim_state, ensemble_size, nonlinearity='linear')
#             compressor_layers.append(layer)
#         self._compressor = nn.Sequential(*compressor_layers).to(device)
#
#         for i_layer in range(n_layers + 1):
#             if i_layer == 0:
#                 layer = EnsembleLayer(dim_state, hidden_size, ensemble_size, nonlinearity)
#             elif 0 < i_layer < n_layers:
#                 layer = EnsembleLayer(hidden_size, hidden_size, ensemble_size, nonlinearity)
#             else:
#                 layer = EnsembleLayer(hidden_size, 2 * dim_state, ensemble_size, nonlinearity='linear')
#             predictor_layers.append(layer)
#         self._predictor = nn.Sequential(*predictor_layers).to(device)
#
#     def setup_normalizer(self, normalizer):
#         self.normalizer = Normalizer(self.dim_state, self.dim_action)
#         # self.normalizer.set_state(normalizer.get_state())
#
#     def get_input(self, state, action):
#         if self.normalizer is None:
#             return state, action
#
#         return self.normalizer.normalize_states(state), self.normalizer.normalize_actions(action)
#
#     def get_target(self, state_delta):
#         if self.normalizer is None:
#             return state_delta
#
#         return self.normalizer.normalize_state_deltas(state_delta)
#
#     def get_output(self, delta_mean, var):
#         if self.normalizer is not None:
#             delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
#             var = self.normalizer.denormalize_state_delta_vars(var)
#
#         return delta_mean, var
#
#     def compressor(self, state_action):
#         x = self._compressor(state_action)
#         mu, log_var = torch.split(x, x.size(-1) // 2, dim=-1)
#         log_var = torch.sigmoid(log_var)
#         # log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var
#
#         return mu, log_var
#
#     def reparametrize(self, mu, log_var):
#         std = log_var.exp().sqrt()
#         eps = std.data.new(std.size()).normal_().to(device)
#         z = mu + eps * std
#
#         return z
#
#     def predictor(self, z):
#         x = self._predictor(z)
#         delta_mean, log_var = torch.split(x, x.size(-1) // 2, dim=-1)
#         log_var = torch.sigmoid(log_var)  # in [0, 1]
#         # log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var
#         var = torch.exp(log_var)  # normal scale, not log
#
#         return delta_mean, var
#
#     def feed_forward(self, state, action):
#         norm_state, norm_action = self.get_input(state, action)
#         state_action = torch.cat((norm_state, norm_action), dim=-1)
#         mu, log_var = self.compressor(state_action)
#         z = self.reparametrize(mu, log_var)
#         delta_mean, var = self.predictor(z)
#         var_ = log_var.exp()
#
#         return mu, var_, z, delta_mean, var
#
#     def forward(self, state, action):
#         states = state.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
#         actions = action.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
#         _, _, _, delta_mean, var = self.feed_forward(states, actions)
#         delta_mean, var = self.get_output(delta_mean, var)
#         # mean = torch.mean(delta_mean, dim=0) + state
#         # next_state_mean, next_action_var = mean.transpose(0, 1), var.transpose(0, 1)
#         # next_state_mean, next_state_var = torch.mean(mean, dim=0), torch.mean(var, dim=0)
#
#         return torch.mean(delta_mean, dim=0), torch.mean(var, dim=0)
#
#     def sample(self, mean, var):
#
#         return Normal(mean, torch.sqrt(var)).sample()
#
#     @staticmethod
#     def log_norm(x, mu, std):
#         """
#         compute the log pdf of x
#
#         :param x: x
#         :param mu: normal distribution with mean mu
#         :param std: normal distribution with standard deviation std
#
#         :return: the log pdf of x.
#         """
#
#         return -0.5 * torch.log(2 * np.pi * std ** 2) - (0.5 * (1 / (std ** 2)) * (x - mu) ** 2)
#
#     def loss(self, states, actions, state_deltas, alpha):
#         """
#         compute renyi divergence: 1 / (1 - alpha) * log((1 / k) * sum(((p(z, x) / q(z)) ** (1 - alpha))
#         p(z, x) = p(x|z) * p(z)
#         in order to stable the expectation computation, using numerical trick:
#             ratio = (1 - alpha)[log p(x, z) - log q(z)]
#             adj = max(ratio)
#             [log (1 / k) * sum(exp(ratio - adj))] + adj
#
#         :param states:
#         :param actions:
#         :param state_deltas:
#         :param alpha:
#
#         :return:
#         """
#         # print(states.size())
#         # exit()
#         states = states.view(-1, self.ensemble_size, self.dim_state).transpose(0, 1)
#         actions = actions.view(-1, self.ensemble_size, self.dim_action).transpose(0, 1)
#         state_deltas = state_deltas.view(-1, self.ensemble_size, self.dim_state).transpose(0, 1)
#         mu0, var0, z, mu1, var1 = self.feed_forward(states, actions)
#         targets = self.get_target(state_deltas)
#
#         q_z = self.log_norm(z, mu0, var0)       # same to compute likelihood, z is similarity targets.
#         prior_z = self.log_norm(z, torch.tensor(0.0), torch.tensor(1.0))        # prior
#         likelihood = self.log_norm(targets, mu1, var1)
#         ratio = (1 - alpha) * (prior_z + likelihood - q_z)
#         const = torch.max(ratio)
#         expectant = torch.exp(ratio - const)
#         renyi_div = - (torch.log(torch.mean(expectant)) + const) / (1 - alpha)
#
#         # print(renyi_div)
#
#         return renyi_div
#
#     def renyi_alpha_div(self, states, actions, delta_states, alpha):
#         states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
#         actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
#         delta_states = delta_states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
#         mean_, log_var_, z, delta_mu, delta_std = self.feed_forward(states, actions)
#
#         q_z = self.log_norm(z, mean_, F.softplus(log_var_).sqrt())
#         prior_z = self.log_norm(z, torch.zeros(mean_.size()).to(device), torch.ones(mean_.size()).to(device))
#         likelihood = self.log_norm(delta_states, delta_mu, delta_std.sqrt())
#
#         ratio = (1 - alpha) * (prior_z + likelihood - q_z)
#         const = torch.max(ratio)
#         renyi = - ((ratio - const).exp().mean().log() + const) / (1 - alpha)
#
#         return renyi / (mean_.size(0))
