import os
import numpy as np
from glob import glob
from pathlib import Path
from matplotlib import pyplot as plt

import os.path as op


def get_data(path, env, algo):
    """one algo, one task"""
    eval_path = glob(str(path / f"{env}_{algo}*.npy"))
    # eval_path = op.join(path, f"{env}_{algo}*.npy")
    eval_data = []

    for data_path in eval_path:
        data = np.load(data_path)
        eval_data.append(data)

    eval_data = np.array(eval_data)
    data_mean = np.mean(eval_data, axis=0)
    data_max = np.max(eval_data, axis=0)
    data_min = np.min(eval_data, axis=0)

    return data_max, data_mean, data_min


def plot_exp(data_max, data_mean, data_min, algo, color, x_limit, y_limit, save_name):
    x = np.arange(len(data_mean)) / len(data_mean)      # the length of the x axis

    plt.style.use('seaborn-darkgrid')      # setting the background of the plot whitegrid 'seaborn-darkgrid'
    plt.plot(x, data_mean, color=color, label=algo)
    plt.fill_between(x, data_max, data_min, facecolor=color, alpha=0.1)

    if save_name is not None:
        plt.xlabel('million steps', fontsize=18)
        plt.ylabel('average return', fontsize=18)
        plt.xlim(x_limit)
        plt.ylim(y_limit[save_name])

        if save_name == 'Humanoid-v2':          # legend in one pic
            plt.legend(loc=2, fontsize=14)
        # plt.legend(loc=2, fontsize=14)        # legend on each pic

        plt.savefig(fname=os.path.join(save_name + ".pdf"), bbox_inches='tight', pad_inches=0.0)
        # print("hhhhhhhhhhh")
        plt.close()


if __name__ == '__main__':
    root_path = Path('KL-vs-Renyi')
    environments = []

    colors = {"DivAC-KL": "#0000ff", "DivAC": "#ff3300"}
    y_limit = {"Ant-v2": [-10, 6000],
               "Walker2d-v2": [-100, 5500],
               "Swimmer-v2": [5, 140],
               "Humanoid-v2": [-100, 5600],
               "HalfCheetah-v2": [-100, 12000],
               "Hopper-v2": [-100, 4000]}
    x_limit = [0, 1]

    for env in os.listdir(root_path):
            env_name = str(env).split("_")[0]
            if env_name not in environments:
                # print("ggggg")
                environments.append(env_name)

                algos = set()       # algo
                for item in glob(str(root_path / f"{env_name}*.npy")):
                    algo_name = str(os.path.basename(item)).split("_")[1]
                    if not algo_name == 'PPO':
                        algos.add(algo_name)

                algos = list(algos)

                for i in range(len(algos)):
                    data_max, data_mean, data_min = get_data(root_path, env_name, algos[i])
                    plot_exp(data_max, data_mean, data_min, algos[i], colors[algos[i]], x_limit, y_limit, 
                             save_name=env_name if (i == len(algos) - 1) else None)
            else:
                continue
            print(env_name)


















