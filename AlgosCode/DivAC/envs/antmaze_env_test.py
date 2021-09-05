import numpy as np
from envs.create_maze_env import create_maze_env
from envs.env_util import EnvWithGoal, success_fn


def run_environment(env_name, episode_length, num_episodes):
    env = EnvWithGoal(create_maze_env(env_name), env_name)
    # env = create_maze_env.create_maze_env(env_name)

    def action_fn(obs):
        action_space = env.action_space
        action_space_mean = (action_space.low + action_space.high) / 2.0
        action_space_magn = (action_space.high - action_space.low) / 2.0
        random_action = (action_space_mean + action_space_magn * np.random.uniform(low=-1.0, high=1.0, size=action_space.shape))
        return random_action
        # return env.action_space.sample()

    rewards = []
    successes = []
    for ep in range(num_episodes):
        rewards.append(0.0)
        successes.append(False)
        obs = env.reset()
        print(obs.shape)
        exit()
        # env.render()
        for _ in range(episode_length):
            obs, reward, done, _ = env.step(action_fn(obs))
            rewards[-1] += reward
            successes[-1] = success_fn(reward)
            if done:
                break
    print(rewards)
    print(successes)


if __name__ == '__main__':
    run_environment("AntMaze", 1000, 10)
