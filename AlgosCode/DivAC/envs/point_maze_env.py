from envs.maze_env import MazeEnv
from envs.point import PointEnv


class PointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
