from envs.maze_env import MazeEnv
from envs.ant import AntEnv


class AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv
