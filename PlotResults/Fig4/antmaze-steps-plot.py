import os
import os.path as op

import numpy as np


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl 


def rename(path):
    for item in os.listdir(path):
        if item.startswith('dac'):
            spl = item.split("_")
            spl[0] = 'DivAC-KL'
            spl[0], spl[1] = spl[1], spl[0]

            dst = "_".join(spl)

            dst = op.join(path, dst)
            src = op.join(path, item)

            os.rename(src, dst)


def step_plot(x, y, color, i, step):
    grey = 1
    white = 0
     
    colors = ['white', 'antiquewhite', 'gainsboro', 'gray'] 
    cmap = mpl.colors.ListedColormap(colors)
    a = np.array([grey, grey, grey, grey, grey, grey, grey, grey,
                  grey, white, white, white, white, white, white, grey,
                  grey, white, white, white, white, white, white, grey,
                  grey, grey, grey, grey, grey, white, white, grey,
                  grey, grey, grey, grey, grey, white, white, grey,
                  grey, white, white, white, white, white, white, grey,
                  grey, white, white, white, white, white, white, grey,
                  grey, grey, grey, grey, grey, grey,grey, grey]).reshape(8,8)

    plt.imshow(a,interpolation = 'nearest',cmap = cmap ,origin = 'up', extent=[-2, 6, -2, 6])

    # i = 0
    plt.plot(x, y, marker='.', color=color, markersize=3)
    # for pos in step_pos:
    #     # i += 1
    #     plt.plot(float(pos[0]), float(pos[1]), marker='o', color=color, markersize=2)
    # print(i)
    #显示右边的栏
    # plt.colorbar(shrink = .92)
    # plt.xlim(-2, 6)
    # plt.ylim(-2, 6)
    # plt.plot(1, 3, 'ro')

    #ignore ticks
    plt.xticks([])
    plt.yticks([])
    
    if i == 4:
        # plt.show()
        plt.savefig(fname=os.path.join(str(step) + ".png"), dpi=400, bbox_inches='tight', pad_inches=0.0)
        plt.close()


def pltimshow():
    grey = 1
    white = 0
     
    colors = ['white', 'antiquewhite', 'gainsboro', 'gray'] 
    cmap = mpl.colors.ListedColormap(colors)
    a = np.array([grey, grey, grey, grey, grey, grey, grey, grey,
                  grey, white, white, white, white, white, white, grey,
                  grey, white, white, white, white, white, white, grey,
                  grey, grey, grey, grey, grey, white, white, grey,
                  grey, grey, grey, grey, grey, white, white, grey,
                  grey, white, white, white, white, white, white, grey,
                  grey, white, white, white, white, white, white, grey,
                  grey, grey, grey, grey, grey, grey,grey, grey]).reshape(8,8)

    plt.imshow(a,interpolation = 'nearest',cmap = cmap ,origin = 'up', extent=[-2, 6, -2, 6])
    # ignore ticks
    plt.xticks([])
    plt.yticks([])



if __name__ == '__main__':
    # pltimshow()
    # plt.show()

    path = './'
    color = ['#00FFFF', '#76EE00', '#8A2BE2', '#EE82EE', '#FFD700']
    steps = [500, 3000, 7000, 10000]
    for _, step in enumerate(steps):
        i = 0
        for item in sorted(os.listdir(path)):
            if item.endswith('pos.npy'):
                x, y = [], []
                res = np.load(op.join(path, item), allow_pickle=True).tolist()

                for pos in res[step]:
                    x.append(float(pos[0]))
                    y.append(float(pos[1]))
                step_plot(x, y, color[i], i, step)
                i += 1









