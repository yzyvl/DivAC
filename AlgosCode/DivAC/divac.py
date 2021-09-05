import os
import torch
import torch.nn.functional as F

from torch import float32 as tf32
from torch.optim import Adam
from utils.utils import soft_update, hard_update
from model.q_policy import QNet, GaussianPolicy, DeterministicPolicy
from model.kl import KLDivModel
from model.renyi import RenyiDivModel


class DivAC(object):
    def __init__(self, d_state, action_space, args):
        self.gamma, self.tau, self.alpha1, self.alpha2 = args.gamma, args.tau, args.alpha1, args.alpha2
        # self.target_update, self.update = args.target_update, 0
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        self.critic = QNet(d_state, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNet(d_state, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if args.d_g == "Deterministic":
            self.policy = DeterministicPolicy(d_state, action_space.shape[0], args.hidden_size, float(action_space.high[0])).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.policy = GaussianPolicy(d_state, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        if args.alg == "DivAC-KL":
            self.div = KLDivModel(d_state, action_space.shape[0], args.hidden_size, self.device).to(self.device)
            self.optim = Adam(self.div.parameters(), lr=args.lr)
        else:
            self.div = RenyiDivModel(d_state, action_space.shape[0], args.hidden_size, self.device).to(self.device)
            self.optim = Adam(self.div.parameters(), lr=args.lr)

    def select_action(self, state, evaluation=False):
        state = torch.tensor(state, dtype=tf32).to(self.device).unsqueeze(0)
        if evaluation:        # test
            _, action = self.policy.sample(state)
        else:
            action, _ = self.policy.sample(state)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        # self.update += 1
        state_b, action_b, reward_b, next_state_b, mask_b = memory.sample(batch_size)

        state_b = torch.tensor(state_b, dtype=tf32).to(self.device)
        action_b = torch.tensor(action_b, dtype=tf32).to(self.device)
        next_state_b = torch.tensor(next_state_b, dtype=tf32).to(self.device)
        reward_b = torch.tensor(reward_b, dtype=tf32).to(self.device).unsqueeze(1)
        mask_b = torch.tensor(mask_b, dtype=tf32).to(self.device).unsqueeze(1)
        self.update_div(self.div, self.optim, state_b, action_b, next_state_b, self.alpha1)
        with torch.no_grad():
            next_state_action, _ = self.policy.sample(next_state_b)
            div = self.div.div(state_b, action_b, next_state_b, self.alpha1)
            q1_next_target, q2_next_target = self.critic_target(next_state_b, next_state_action)
            min_qf_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha2 * div
            next_q_value = reward_b + mask_b * self.gamma * min_qf_next_target

        q1, q2 = self.critic(state_b, action_b)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)

        actions_pi, _ = self.policy.sample(state_b)
        q1_pi, q2_pi = self.critic(state_b, actions_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        div_pi = self.div.div(state_b, actions_pi, next_state_b, self.alpha1)
        policy_loss = ((self.alpha2 * div_pi) - min_q_pi).mean()

        # self.update_div(self.div, self.optim, state_b, action_b, next_state_b, self.alpha1)

        self.critic_optim.zero_grad()
        q1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        q2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)

        # if self.update % self.target_update == 0:
        #     soft_update(self.critic_target, self.critic, self.tau)

    def update_div(self, model, optim, state, action, delta_state, alpha):
        optim.zero_grad()
        loss = model.loss(state, action, delta_state, alpha)
        loss.backward()
        optim.step()
