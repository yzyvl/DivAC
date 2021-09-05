import argparse
import itertools
import random
import gym
import os
import numpy as np
import logging
import torch

from datetime import datetime as dt
from divac import DivAC
from utils.replay_memory import ReplayMemory

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


@torch.no_grad()
def _test(env, agent, logging, i_episode):
    avg_reward, episodes = 0., 10
    max_r, min_r = -99999, 99999
    for _ in range(episodes):
        obs = env.reset()
        test_reward, _done = 0, False
        while not _done:
            with torch.no_grad():
                action = agent.select_action(obs, evaluation=True)
            next_obs, r, _done, _ = env.step(action)
            test_reward += r
            obs = next_obs
        if max_r < test_reward:
            max_r = test_reward
        if min_r > test_reward:
            min_r = test_reward
        avg_reward += test_reward
    avg_reward /= episodes

    print("----------------------------------------")
    print(f"Training Episode: {i_episode}, Avg: {avg_reward}, Max: {max_r}, Min: {min_r}")
    logging.info(f"Training Episode: {i_episode}, Avg: {avg_reward}, Max: {max_r}, Min: {min_r}")
    print("----------------------------------------")

    return avg_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='HalfCheetah-v2',
                        choices=['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'HumanoidStandup-v2', 'Humanoid-v2',
                                 'Swimmer-v2', 'Reacher-v2', 'Walker2d-v2', 'InvertedDoublePendulum-v2'])
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-lr', type=float, default=0.0003)
    parser.add_argument('-alpha1', type=float, default=0.3)
    parser.add_argument('-alpha2', type=float, default=0.5)
    parser.add_argument('-tau', type=float, default=0.003)
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-num_steps', type=int, default=1000000)
    parser.add_argument('-hidden_size', type=int, default=256)
    parser.add_argument('-start_steps', type=int, default=5000)
    # parser.add_argument('-target_update', type=int, default=1)
    # parser.add_argument('-updates_per_step', type=int, default=1)
    parser.add_argument('-capacity', type=int, default=1000000)
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-log', type=str, default='./results')
    parser.add_argument('-test_steps', type=int, default=5000)
    parser.add_argument('-alg', type=str, default='DivAC-KL')
    parser.add_argument('-d_g', type=str, default="Stochastic")       # Deterministic

    args = parser.parse_args()
    args.seed = random.randint(1, 10000)
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

    evaluations = []

    d_state, d_action = env.observation_space.shape[0], env.action_space.shape[0]
    memory = ReplayMemory(args.capacity)
    agent = DivAC(d_state, env.action_space, args)

    # Training Loop
    total_numsteps = 0

    for i_episode in itertools.count(1):
        episode_reward, episode_pseudo = 0, 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                # for i in range(args.updates_per_step):
                agent.update_parameters(memory, args.batch_size)

            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)
            state = next_state

            if total_numsteps % args.test_steps == 0:
                avg_reward = _test(env, agent, logging, i_episode)
                evaluations.append(avg_reward)

        if total_numsteps > args.num_steps:
            break

        print(f"Total steps: {total_numsteps}, episode steps: {episode_steps}, episode reward: {round(episode_reward, 2)}")
    np.save(f"{filename[:-4]}", evaluations)
    env.close()
