import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class Bnn(object):
    # * this network is utilized to generate the parameters(two parameters:mu & sigma)
    def __init__(self, observation_dim, action_dim, hidden_dim, max_logvar, min_logvar):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_logvar = max_logvar
        self.min_logvar = min_logvar

        self.w1_mu = torch.zeros(self.observation_dim + self.action_dim, self.hidden_dim)
        self.b1_mu = torch.zeros(self.hidden_dim)
        self.w2_mu = torch.zeros(self.hidden_dim, self.observation_dim * 2)
        self.b2_mu = torch.zeros(self.observation_dim * 2)

        self.w1_var = torch.zeros(self.observation_dim + self.action_dim, self.hidden_dim)
        self.b1_var = torch.zeros(self.hidden_dim)
        self.w2_var = torch.zeros(self.hidden_dim, self.observation_dim * 2)
        self.b2_var = torch.zeros(self.observation_dim * 2)

        self.w1_size = np.prod(self.w1_mu.size())
        self.b1_size = np.prod(self.b1_mu.size())
        self.w2_size = np.prod(self.w2_mu.size())
        self.b2_size = np.prod(self.b2_mu.size())
        self.net_parameter_num = self.w1_size + self.b1_size + self.w2_size + self.b2_size

    def set_params(self, param_mu, param_rho):
        self.w1_mu = param_mu[0: self.w1_size].view(self.w1_mu.size())
        self.b1_mu = param_mu[self.w1_size: self.w1_size + self.b1_size].view(self.b1_mu.size())
        self.w2_mu = param_mu[self.w1_size + self.b1_size: self.w1_size + self.b1_size + self.w2_size].view(
            self.w2_mu.size())
        self.b2_mu = param_mu[self.w1_size + self.b1_size + self.w2_size:].view(self.b2_mu.size())

        w1_rho = param_rho[0: self.w1_size].view(self.w1_var.size())
        b1_rho = param_rho[self.w1_size: self.w1_size + self.b1_size].view(self.b1_var.size())
        w2_rho = param_rho[self.w1_size + self.b1_size: self.w1_size + self.b1_size + self.w2_size].view(
            self.w2_var.size())
        b2_rho = param_rho[self.w1_size + self.b1_size + self.w2_size:].view(self.b2_var.size())

        self.w1_var = (1 + torch.exp(w1_rho)).log().pow(2)
        self.b1_var = (1 + torch.exp(b1_rho)).log().pow(2)
        self.w2_var = (1 + torch.exp(w2_rho)).log().pow(2)
        self.b2_var = (1 + torch.exp(b2_rho)).log().pow(2)

    def linear(self, w_mu, b_mu, w_var, b_var, x):
        mean = x @ w_mu + b_mu
        variance = x.pow(2) @ w_var + b_var
        # * Local Reparameterization Trick
        noise = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(variance)).sample()
        output = mean + variance.pow(0.5) * noise
        return output

    def infer(self, observation, action):
        input = torch.cat([observation, action], 1)
        x = F.relu(self.linear(self.w1_mu, self.b1_mu, self.w1_var, self.b1_var, input))
        x = F.relu(self.linear(self.w2_mu, self.b2_mu, self.w2_var, self.b2_var, x))
        mean, logvar = x[:, : self.observation_dim], x[:, self.observation_dim:]
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def calc_log_likelihood(self, next_observations, actions, observations):
        # * calculate the log-likelihood term and the KL divergence of the loss function is ZERO\
        # print(next_observations.shape, actions.shape, observations.shape)
        # exit()
        next_mean, next_logvar = self.infer(observations, actions)
        # * it assumes that weight distribution q(theta; phi) is given by the fully factorized Gaussian distribution
        # * so the covariance matrix is diagonal and it reduces the computation
        log_likelihood = - 0.5 * ((next_observations - next_mean).pow(2) * (- next_logvar).exp() + next_logvar).sum(
            1) - 0.5 * self.observation_dim * np.log(2 * np.pi)
        return log_likelihood


class Vime(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size, min_logvar, max_logvar, learning_rate,
                 kl_buffer_capacity, lamda, update_iteration, batch_size, eta):
        super(Vime, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.learning_rate = learning_rate
        self.kl_buffer_capacity = kl_buffer_capacity
        self.lamda = lamda
        self.update_iteration = update_iteration
        self.batch_size = batch_size
        self.eta = eta

        self.dynamics_model = Bnn(self.observation_dim, self.action_dim, self.hidden_size, self.max_logvar, self.min_logvar)
        self.param_mu = nn.Parameter(torch.zeros(self.dynamics_model.net_parameter_num))
        self.param_rho = nn.Parameter(torch.zeros(self.dynamics_model.net_parameter_num))
        self.dynamics_model.set_params(self.param_mu, self.param_rho)
        self.optimizer = torch.optim.Adam([self.param_mu, self.param_rho], lr=self.learning_rate)
        self.kl_buffer = deque(maxlen=self.kl_buffer_capacity)

    def calc_info_gain(self, observation, action, next_observation):
        self.dynamics_model.set_params(self.param_mu, self.param_rho)
        log_likelihood = self.dynamics_model.calc_log_likelihood(torch.FloatTensor(np.expand_dims(next_observation, 0)),
                                                                 torch.FloatTensor(np.expand_dims(action, 0)),
                                                                 torch.FloatTensor(np.expand_dims(observation, 0)))
        log_likelihood = log_likelihood.mean()
        self.optimizer.zero_grad()
        (- log_likelihood).backward()
        nabla = torch.cat([self.param_mu.grad.detach(), self.param_rho.grad.detach()])
        H = self.calc_hessian()
        info_gain = (self.lamda ** 2 / 2 * nabla.pow(2) * H.pow(-1)).sum().detach()
        return info_gain.item()

    def calc_hessian(self):
        # * calculate the hessian matrix of the KL term and ignore the hessian matrix of the log-likelihood term
        H_mu = (1 + torch.exp(self.param_rho)).log().pow(-2).detach()
        H_rho = (1 + torch.exp(self.param_rho)).log().pow(-2) * 2 * torch.exp(2 * self.param_rho) * (
                    1 + torch.exp(self.param_rho)).pow(-2)
        H_rho = H_rho.detach()
        # * find KL divergence partial guide to mu and rho
        H = torch.cat([H_mu, H_rho], -1).detach()
        return H

    def calc_kl_div(self, prev_mu, prev_var):
        # * calculate the KL divergence term
        var = (1 + torch.exp(self.param_rho)).log().pow(2)
        kl_div = 0.5 * ((var / prev_var) + prev_var.log() - var.log() + (prev_mu - self.param_mu).pow(
            2) / prev_var).sum() - 0.5 * len(self.param_mu)
        return kl_div

    def update(self, buffer):
        # * maximize the elbo
        elbo = 0
        prev_mu, prev_var = self.param_mu.detach(), (1 + torch.exp(self.param_rho.detach())).log().pow(2)
        for i in range(self.update_iteration):
            observations, actions, _, next_observations, _ = buffer.sample(self.batch_size)
            observations = torch.FloatTensor(observations)
            actions = torch.FloatTensor(actions)
            next_observations = torch.FloatTensor(next_observations)
            self.dynamics_model.set_params(self.param_mu, self.param_rho)

            log_likelihood = self.dynamics_model.calc_log_likelihood(next_observations, actions, observations).mean()
            div_kl = self.calc_kl_div(prev_mu, prev_var)
            elbo = log_likelihood - div_kl

            self.optimizer.zero_grad()
            (- elbo).backward(retain_graph=True)
            self.optimizer.step()

        return elbo

    def store_kl(self, info_gains):
        self.kl_buffer.append(np.median(info_gains))

    def calc_curiosity_reward(self, rewards, info_gains):
        if len(self.kl_buffer) == 0:
            relative_gains = info_gains
        else:
            # * prevent the mean of the previous kl to be ZERO
            relative_gains = info_gains / np.mean(self.kl_buffer) if np.mean(self.kl_buffer) != 0 else info_gains
        return rewards + self.eta * relative_gains
