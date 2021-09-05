from envs.ant_maze_env import AntMazeEnv
from envs.point_maze_env import PointMazeEnv


def create_maze_env(env_name=None, top_down_view=False):
    n_bins = 0
    manual_collision = False
    if env_name.startswith('Ego'):
        n_bins = 8
        env_name = env_name[3:]
    if env_name.startswith('Ant'):
        cls = AntMazeEnv
        env_name = env_name[3:]
        maze_size_scaling = 8
    elif env_name.startswith('Point'):
        cls = PointMazeEnv
        manual_collision = True
        env_name = env_name[5:]
        maze_size_scaling = 4
    else:
        assert False, 'unknown env %s' % env_name

    maze_id = None
    observe_blocks = False
    put_spin_near_agent = False
    if env_name == 'Maze':
        maze_id = 'Maze'
    elif env_name == 'Push':
        maze_id = 'Push'
    elif env_name == 'Fall':
        maze_id = 'Fall'
    elif env_name == 'Block':
        maze_id = 'Block'
        put_spin_near_agent = True
        observe_blocks = True
    elif env_name == 'BlockMaze':
        maze_id = 'BlockMaze'
        put_spin_near_agent = True
        observe_blocks = True
    else:
        raise ValueError('Unknown maze environment %s' % env_name)

    gym_mujoco_kwargs = {'maze_id': maze_id,
                         'n_bins': n_bins,
                         'observe_blocks': observe_blocks,
                         'put_spin_near_agent': put_spin_near_agent,
                         'top_down_view': top_down_view,
                         'manual_collision': manual_collision,
                         'maze_size_scaling': maze_size_scaling}
    gym_env = cls(**gym_mujoco_kwargs)
    gym_env.reset()
    # wrapped_env = gym_wrapper.GymWrapper(gym_env)
    return gym_env


if __name__ == '__main__':
    env = create_maze_env(env_name='AntMaze')
    # print(env)
    # env = gym.make('AntMaze-v2')
    # env.render()
    state = env.reset()
    # env.render()
    # env.render()

    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        print(reward)

        if reward > -5.:
            break
