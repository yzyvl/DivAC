import gym
import os
import numpy as np
#
from gym import register
from envs import AntEnv
from envs.ant2 import rate_buffer

# a = [0.2857, 0.4286, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.8571, 0.8571, 0.8571]
# b = [0.2857, 0.4286, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714]
# c = [0.2857, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714]
# d = [0.4286, 0.4286, 0.4286, 0.4286, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714]
# e = [0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.5714, 0.7143, 0.7143, 0.7143, 0.8571]
#
# np.save('ant_results/AntMaze_Random_7799', a)
# np.save('ant_results/AntMaze_Random_80943', b)
# np.save('ant_results/AntMaze_Random_473894', c)
# np.save('ant_results/AntMaze_Random_237445', d)
# np.save('ant_results/AntMaze_Random_790834', e)
#
# exit()


register(id='AntMaze-v1', entry_point=AntEnv)

seeds = [7834, 892344, 8924, 2340, 327476]

env = gym.make('AntMaze-v1')

for seed in seeds:
    env.seed(seed)
    np.random.seed(seed)

    evaluations = []

    # for step in [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 100000]:
    max_cov = 0
    # for _ in range(10):
    e_buffer = []
    obs = env.reset()
    # e_buffer = [obs]
    for i in range(10000):
        e_buffer.append(obs)
        action = env.action_space.sample()
        next_obs, _, _, _ = env.step(action)
        obs = next_obs
    for step in [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 100000]:
        cov = rate_buffer(e_buffer[:step])
        evaluations.append(round(cov/7, 4))

    print(evaluations)
    # exit()
