import copy
import re
import string
from pprint import pprint

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation
from tqdm import tqdm
from numba import njit

# %%

directions = np.loadtxt('input', dtype=object).item()
jets = np.array([1 if jet == '>' else -1 for jet in directions])

# %%

shapes = [[[0, 1, 2, 3, 3], [0, 0, 0, 0, 0]],
          [[0, 1, 1, 1, 2], [1, 0, 1, 2, 1]],
          [[0, 1, 2, 2, 2], [0, 0, 0, 1, 2]],
          [[0, 0, 0, 0, 0], [0, 1, 2, 3, 3]],
          [[0, 0, 1, 1, 1], [0, 1, 0, 1, 1]]]
shapes = np.array(shapes)


for shape in shapes:
    s = np.zeros((4, 4))
    s[tuple(shape)] = 1
    plt.imshow(s.T, origin='lower')
    plt.show()

# %%


@njit
def run_simulation(shapes, jets, stopped_rocks):
    height = 500
    width = 7
    height_thresh = 40
    height_move = 30
    grid_height = 0
    grid = np.zeros((width, height))
    grid[:, 0] = 1
    jet_i = 0
    max_ys = []
    tower_heights = []
    for i in range(stopped_rocks):
        if i % 10_000 == 0:
            print(i, 'out of', stopped_rocks)
        shape = shapes[i % len(shapes)]
        max_y = [(np.where(grid == 1)[1])[np.where(grid == 1)[0] == iter_x].max() for iter_x in range(width)]

        if (np.array(max_y) > height_thresh).all():
            grid[:, :-height_move] = grid[:, height_move:]
            grid_height += height_move
            max_y = [(np.where(grid == 1)[1])[np.where(grid == 1)[0] == iter_x].max() for iter_x in range(width)]

        tower_heights.append([i, grid_height + max(max_y)])

        rock_x = 2
        rock_y = max(max_y) + 4
        max_ys.append([i % len(shapes)] + max_y)  # before rock falls, without jet_i index
        # max_ys.append([i % len(shapes), jet_i] + max_y)  # before rock falls, with jet_i index

        # grid[np.array(shape[0]) + rock_x, np.array(shape[1]) + rock_y] = 1
        # plt.imshow(grid[:, :10].T, origin='lower')
        # plt.show()
        # grid = np.zeros((width, 10_000), dtype=int)
        # grid[:, 0] = 1

        while True:
            # jet
            rock_x += jets[jet_i % len(jets)]
            cord_x = [rock_x + x for x in shape[0]]
            if (max(cord_x) >= width) or (min(cord_x) < 0) or np.array([int(grid[rock_x + x, rock_y + y]) for x, y in shape.T]).sum() > 0:
                rock_x -= jets[jet_i % len(jets)]
            jet_i += 1

            # falling
            rock_y -= 1
            check_grid = [grid[rock_x + x, rock_y + y] for x, y in shape.T]
            if np.array(check_grid).sum() > 0:
                # hit bottom, place rock and continue with next rock
                rock_y += 1
                for x, y in shape.T:
                    grid[x + rock_x, y + rock_y] = 1
                break

        # plt.imshow(grid[:, :10].T, origin='lower')
        # plt.show()

    max_ys = np.array(max_ys)
    return grid, grid_height, max_ys, tower_heights

# %%


stopped_rocks = 50_000
grid, grid_height, max_ys, tower_heights = run_simulation(shapes, jets, stopped_rocks)
width = grid.shape[0]
max_y = [(np.where(grid == 1)[1])[np.where(grid == 1)[0] == iter_x].max() for iter_x in range(width)]
print(f'tower height : {max_y} --> {grid_height + max(max_y)}')
plt.figure(figsize=(10, 10))
# plt.imshow(grid[:, :20].T, origin='lower')
plt.imshow(grid[:, max(max_y) - 20:max(max_y) + 2].T, origin='lower')
plt.show()

# %%


def get_tower_height(stopped_rocks, max_ys, shapes, tower_heights):
    last_shape = (stopped_rocks - 1) % len(shapes)
    max_ys[:, 2:] = max_ys[:, 2:] - max_ys[:, 2:].min(axis=1)[np.newaxis].T
    max_ys_shape = max_ys[max_ys[:, 0] == last_shape]
    u, i, c = np.unique(max_ys_shape, return_counts=True, return_index=True, axis=0)
    last_unique = np.where(np.all(max_ys == max_ys_shape[i[-1]], axis=1))[0][0]
    periods = np.diff(np.where(np.all(max_ys == max_ys_shape[i[-1]], axis=1))[0])
    period = np.unique(periods).sum()
    print('start index:', last_unique)
    print('period:', period)
    init_config = last_unique + (stopped_rocks - last_unique) % period
    height_increase = tower_heights[init_config + period][1] - tower_heights[init_config][1]
    tower_height = tower_heights[init_config][1] + height_increase * (stopped_rocks - init_config) / period
    assert int(tower_height) == tower_height
    return int(tower_height)

# %%


for stopped_rocks in [2022, 1_000_000_000_000]:
    print(f'solution for {stopped_rocks} stopped rocks:', get_tower_height(stopped_rocks, max_ys, shapes, tower_heights))

# 3147
# 1532163742758
# %%
