import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import os
import logging
from sac import SAC
from replay_memory import ReplayMemory


def test_(env_name, agent, seed):
    env = gym.make(env_name)
    avg_reward, episodes = 0., 10
    for _ in range(episodes):
        s = env.reset()
        episode_reward_ = 0
        done = False
        while not done:
            a = agent.select_action(s, evaluate=True)
            # print(a)
            next_s, reward, done, _ = env.step(a)
            episode_reward_ += reward
            s = next_s
        avg_reward += episode_reward_
    avg_reward /= episodes

    print("----------------------------------------")
    print("Evaluation over: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")
    env.close()

    return avg_reward


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="HalfCheetah-v2")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False)
parser.add_argument('--seed', type=int, default=3543)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=1000001)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--updates_per_step', type=int, default=1)
parser.add_argument('--start_steps', type=int, default=10000)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--replay_size', type=int, default=1000000)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('-log', type=str, default='./results')
parser.add_argument('-test_steps', type=int, default=5000)
parser.add_argument('-alg', type=str, default='rac-tsallis')
args = parser.parse_args()

# Environment
env = gym.make(args.env)
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.log):
    os.makedirs(args.log)

print(args)

filename = os.path.join(args.log, "%s_%s_%s.txt" % (args.alg, args.env, args.seed))
logging.basicConfig(level=logging.INFO, filename=filename)
logging.info(args)

evaluations = []

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

for t in range(int(args.num_steps)):
    episode_timesteps += 1

    if args.start_steps > t:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state)

    if len(memory) > args.batch_size:
        # Number of updates per step in environment
        for i in range(args.updates_per_step):
            # Update parameters of all the networks
            agent.update_parameters(memory, args.batch_size)

    # Perform action
    next_state, reward, done, _ = env.step(action)
    mask = 1 if episode_timesteps == env._max_episode_steps else float(not done)
    # Store data in replay buffer
    memory.push(state, action, reward, next_state, mask)  
    state = next_state
    episode_reward += reward

    if done:
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        logging.info(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1 

    # Evaluate episode
    if (t + 1) % args.test_steps == 0:
        te_rewards = test_(args.env, agent, args.seed)
        evaluations.append(te_rewards)  # env_name, agent, train_numsteps
        logging.info("Test steps: {}, Avg. Reward: {}".format(t + 1, round(te_rewards, 2)))
np.save(f"{filename[:-4]}", evaluations)
env.close()
