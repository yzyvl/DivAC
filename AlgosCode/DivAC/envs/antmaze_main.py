import argparse
import datetime
import gym
import os
import numpy as np
import logging
import torch

from kl_div_ac import DAC as KL_DAC
from div_ac import DAC as R_DAC
from utils.replay_memory import ReplayMemory
from envs.create_maze_env import create_maze_env
from envs.env_util import EnvWithGoal
from utils.normalizer import Normalizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='AntMaze')
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-lr', type=float, default=0.0003)
    parser.add_argument('-alpha1', type=float, default=0.9)
    parser.add_argument('-alpha2', type=float, default=0.2)
    parser.add_argument('-tau', type=float, default=0.005)
    parser.add_argument('-seed', type=int, default=976)
    parser.add_argument('-batch_size', type=int, default=129)
    parser.add_argument('-num_steps', type=int, default=10000000)
    parser.add_argument('-hidden_size', type=int, default=256)
    parser.add_argument('-start_steps', type=int, default=5000)
    parser.add_argument('-target_update', type=int, default=1)
    parser.add_argument('-updates_per_step', type=int, default=2)
    parser.add_argument('-capacity', type=int, default=1000000)
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-log', type=str, default='./results')
    parser.add_argument('-test_steps', type=int, default=5000)
    parser.add_argument('-t_num_episodes', type=int, default=10)
    parser.add_argument('-alg', type=str, default='rdac')
    parser.add_argument('-max_episode_steps', type=int, default=1000)

    args = parser.parse_args()

    env = EnvWithGoal(create_maze_env(args.env), args.env)
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

    d_state, d_action = env.reset().shape[0], env.action_space.shape[0]
    max_action = float(env.action_space.high[0])        # 30.0

    normalize = Normalizer(d_state, d_action)
    memory = ReplayMemory(args.capacity)

    if args.alg == "kldac":
        agent = KL_DAC(normalize, d_state, env.action_space, max_action, args)
    elif args.alg == "rdac":
        agent = R_DAC(normalize, d_state, env.action_space, max_action, args)
    else:
        raise NotImplementedError("Not exist this algorithm")

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    ant_position = []

    for t in range(int(args.num_steps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if args.start_steps > t:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        mask = 1 if episode_timesteps == args.max_episode_steps else float(not done)
        # Store data in replay buffer
        memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):
                agent.update_param(memory, args.batch_size)

        if episode_timesteps == args.max_episode_steps or reward > -2.0:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            logging.info(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # Evaluate episode
        if (t + 1) % args.test_steps == 0:
            average_suc = 0.
            average_reward = 0
            with torch.no_grad():
                for ep in range(args.t_num_episodes):
                    # num = 0
                    env = EnvWithGoal(create_maze_env(args.env), args.env)
                    obs = env.reset()
                    last_reward = 0
                    for j in range(args.max_episode_steps):
                        action = agent.select_action(obs)
                        next_obs, r, done, _ = env.step(action)
                        average_reward += r
                        # num += 1
                        if r > -2.0:
                            average_suc += 1
                            break
                        obs = next_obs
                    # print(num)
                average_suc /= args.t_num_episodes
                average_reward = round(average_reward / args.t_num_episodes, 2)

            evaluations.append(average_suc)
            print("Test steps: {}, Avg. Reward: {}, Avg. success: {}".format(t + 1, average_reward, average_suc))
            logging.info("Test steps: {}, Avg. Reward: {}, Avg. success: {}".format(t + 1, average_reward, average_suc))
    np.save(f"{filename[:-4]}", evaluations)
