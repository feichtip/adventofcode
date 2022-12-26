import copy
import re
import string
from pprint import pprint

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%

part2 = True

paths = pd.read_csv('input', header=None, sep=r'\n', engine='python')[0].values
paths = [np.array([xy.split(',') for xy in re.findall(r'([\w,]+)', path)]).astype(int) for path in paths]

# for p, path in enumerate(paths):
#     paths[p][:, 0] = path[:, 0] - 450
#     plt.plot(*path.T)
# plt.show()

grid = np.zeros((1000, 200), dtype=int)
for p, path in enumerate(paths):
    for start, stop in zip(path[1:, :], path[:-1, :]):
        i_min, i_max = sorted([start[0], stop[0]])
        j_min, j_max = sorted([start[1], stop[1]])
        grid[slice(i_min, i_max + 1), slice(j_min, j_max + 1)] = 1
    plt.plot(*path.T)
plt.show()


if part2:
    grid[:, np.where(grid)[1].max() + 2] = 1

# %%

n_stopped = 0

finished = False
while not finished:
    sand_x = 500
    sand_y = 0

    # end condition part 2
    if grid[(sand_x, sand_y)] == 2:
        print('source blocked')
        finished = True
        break

    while True:
        # print(sand_x, sand_y)
        targets = [(sand_x, sand_y + 1), (sand_x - 1, sand_y + 1), (sand_x + 1, sand_y + 1)]

        # end condition part 1
        if (sand_y >= grid.shape[1] - 1):
            print('falling out of frame')
            grid[(sand_x, sand_y)] = 2
            finished = True
            break

        if np.all([grid[target] != 0 for target in targets]):  # sand stopps
            grid[(sand_x, sand_y)] = 2
            n_stopped += 1
            break

        for target in targets:
            if grid[target] == 0:
                sand_x, sand_y = target
                break

    # break

# %%

print(f'stopped sand: {n_stopped}')
plt.figure(figsize=(17, 17))
plt.imshow(grid.T)
plt.plot([500], [0], marker='x', ms=10, color='red')
plt.plot([500, 500], [0, 10], marker='', ms=10, color='red', ls=':')
if part2:
    plt.xlim(250, 750)
else:
    plt.xlim(450, 550)

# %%
