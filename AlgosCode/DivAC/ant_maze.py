import argparse
import gym
import os
import numpy as np
import logging
import torch

from datetime import datetime as dt
from gym import register
from utils.replay_memory import ReplayMemory
from envs import AntEnv
from envs.ant2 import rate_buffer
from divac import DivAC

register(id='AntMaze-v1', entry_point=AntEnv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='AntMaze-v1')
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-lr', type=float, default=0.0003)
    parser.add_argument('-alpha1', type=float, default=0.9)
    parser.add_argument('-alpha2', type=float, default=0.5)
    parser.add_argument('-tau', type=float, default=0.005)
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-num_steps', type=int, default=1000000)
    parser.add_argument('-hidden_size', type=int, default=256)
    parser.add_argument('-start_steps', type=int, default=5000)
    parser.add_argument('-target_update', type=int, default=1)
    parser.add_argument('-updates_per_step', type=int, default=1)
    parser.add_argument('-capacity', type=int, default=1000000)
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-log', type=str, default='./results')
    parser.add_argument('-test_steps', type=int, default=5000)
    parser.add_argument('-alg', type=str, default='DivAC-KL')
    parser.add_argument('-d_g', type=str, default="Stochastic")       # Deterministic

    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    print(args)

    filename = os.path.join(args.log, "%s_%s-%s_%s.txt" % (args.env, args.alg, args.d_g, dt.now()))
    logging.basicConfig(level=logging.INFO, filename=filename)
    logging.info(args)

    d_state, d_action = env.observation_space.shape[0], env.action_space.shape[0]
    memory = ReplayMemory(args.capacity)
    agent = DivAC(d_state, env.action_space, args)

    state, done = env.reset(), False
    episode_timesteps = 0
    eval_coverages, eval_pos = [], {}

    for t in range(int(args.num_steps)):

        if args.start_steps > t:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action)
        mask = 1 if episode_timesteps == args.num_steps else float(not done)
        memory.push(state, action, reward, next_state, mask)
        state = next_state

        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):
                agent.update_parameters(memory, args.batch_size)

        if (t + 1) % args.test_steps == 0 or (t + 1) == 500:
            if (t + 1) not in eval_pos:
                eval_pos[t + 1] = []
            with torch.no_grad():
                state_pos, max_cov, num = {}, -1, -1
                for e_i in range(10):
                    state_pos[e_i] = []
                    obs = env.reset()
                    e_buffer = []
                    for i in range(t + 1):
                        e_buffer.append(obs)
                        state_pos[e_i].append((obs[2].item(), obs[3].item()))
                        a = agent.select_action(obs)
                        next_obs, _, _, _ = env.step(a)
                        obs = next_obs
                    cov = rate_buffer(e_buffer)

                    # recorder the episode of max coverage
                    if cov >= max_cov:
                        max_cov = cov
                        num = e_i

                # saving max episode
                max_ep = round(max_cov / 7, 4)
                eval_coverages.append(max_ep)
                eval_pos[t + 1] = state_pos[num]
                print(f'Env steps: {t + 1}, Coverage rate: {max_ep}')

    np.save(f"{filename[:-4]}", eval_coverages)
    np.save(f"{filename[:-4] + '_pos'}", eval_pos)
