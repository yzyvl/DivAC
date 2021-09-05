import numpy as np
import create_maze_env


def get_goal_sample_fn(env_name):
    if env_name == 'AntMaze':
        # NOTE: When evaluating (i.e. the metrics shown in the paper,
        # we use the commented out goal sampling function.  The uncommented
        # one is only used for training.
        # return lambda: np.array([0., 16.])
        return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntPush':
        return lambda: np.array([0., 19.])
    elif env_name == 'AntFall':
        return lambda: np.array([0., 27., 4.5])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    if env_name == 'AntMaze':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntPush':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'


def success_fn(last_reward):
    return last_reward > -5.0


class EnvWithGoal(object):
    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.goal_sample_fn = get_goal_sample_fn(env_name)
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None

    def reset(self):
        obs = self.base_env.reset()
        self.goal = self.goal_sample_fn()
        return np.concatenate([obs, self.goal])

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.reward_fn(obs, self.goal)
        return np.concatenate([obs, self.goal]), reward, done, info

    @property
    def action_space(self):
        return self.base_env.action_space


def run_environment(env_name, episode_length, num_episodes):
    env = EnvWithGoal(create_maze_env.create_maze_env(env_name), env_name)
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
