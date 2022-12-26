import copy
import random
import re
import string
from ast import literal_eval
from pprint import pprint

import awkward as ak
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from sympy import Eq, solve, symbols
from tqdm import tqdm

# %%


# file = np.loadtxt('test_input', dtype=object, delimiter='\n', comments=None)
file = np.loadtxt('input', dtype=object, delimiter='\n', comments=None)

map_size = (len(file[:-1]) + 2, max([len(row) for row in file[:-1]]) + 2)
map = np.full(map_size, -1, dtype=int)
for i, row in enumerate(file[:-1]):
    map[i + 1, 1:len(row) + 1] = [0 if sign == '.' else (1 if sign == '#' else -1) for sign in row]

plt.imshow(map)

# %%

dir_movement = {0: np.array([0, 1]),
                1: np.array([1, 0]),
                2: np.array([0, -1]),
                3: np.array([-1, 0])}
dir_wrap = {0: ('(pos[0], slice(None))', 0, 1),
            1: ('(slice(None), pos[1])', 0, 0),
            2: ('(pos[0], slice(None))', -1, 1),
            3: ('(slice(None), pos[1])', -1, 0)}
edge_id = {0: (0, slice(51, 101)),
           1: (0, slice(101, 151)),
           2: (slice(1, 51), 151),
           3: (51, slice(101, 151)),
           4: (slice(51, 101), 101),
           5: (slice(150, 100, -1), 101),
           6: (151, slice(51, 101)),
           7: (slice(151, 201), 51),
           8: (201, slice(1, 51)),
           9: (slice(151, 201), 0),
           10: (slice(101, 151), 0),
           11: (100, slice(1, 51)),
           12: (slice(51, 101), 50),
           13: (slice(50, 0, -1), 50)}

# Facing is 0 for right (>), 1 for down (v), 2 for left (<), and 3 for up (^)
# how orientation changes when moving from current edge (key) to target edge
edge_rot = {0: 0, 1: 3, 2: 2, 3: 2, 4: 3, 5: 2, 6: 2, 7: 3, 8: 1, 9: 1, 10: 0, 11: 0, 12: 1, 13: 0}
edge_mapping = np.array([[0, 9], [1, 8], [2, 5], [3, 4], [6, 7], [10, 13], [11, 12]])
index_map = np.stack(np.meshgrid(np.arange(202), np.arange(152)), axis=2).transpose(1, 0, 2)

# for key, value in edge_id.items():
#     map[value] = 4

plt.figure(figsize=(12, 12))
plt.imshow(map)
plt.show()

# %%


part2 = True

dir = 0
pos_x = np.where(map[1] == 0)[0][0]
pos_y = 1
pos = np.array([pos_y, pos_x])
positions = [tuple(pos)]

for cmd in tqdm(re.findall(r'\d+|\D+', file[-1])):
    # print(cmd, dir, pos)
    if cmd == 'R':
        dir = (dir + 1) % 4
    elif cmd == 'L':
        dir = (dir - 1) % 4
    else:
        for _ in range(int(cmd)):
            target_idx = pos + dir_movement[dir]
            target_val = map[tuple(target_idx)]
            if target_val == 0:
                pos = target_idx
            elif target_val == -1:
                # wrap around
                if not part2:
                    sl, idx, goto_i = dir_wrap[dir]
                    goto = np.where(map[eval(sl)] != -1)[0][idx]
                    if map[eval(sl)][goto] == 0:
                        pos[goto_i] = goto
                    else:
                        break
                else:
                    current_edge = [key for key, value in edge_id.items()
                                    if (target_idx == index_map[value]).all(axis=1).any()]
                    if len(current_edge) > 1:
                        # does not account for [11, 12]
                        print(current_edge, current_edge[abs(1 - dir)], pos)
                        current_edge = current_edge[abs(1 - dir)]
                        # assert False
                    else:
                        current_edge = current_edge[0]
                    edge_idx = np.where(current_edge == edge_mapping)
                    target_edge = edge_mapping[edge_idx[0].item(), abs(edge_idx[1].item() - 1)]

                    # index of accompanying border
                    wrap_idx = index_map[edge_id[target_edge]][(index_map[edge_id[current_edge]] == target_idx).all(axis=1)][0]
                    # print(wrap_idx)
                    wrap_idx += dir_movement[edge_rot[current_edge]]  # move into cube
                    if map[tuple(wrap_idx)] == 0:
                        # update pos and dir if space is empty
                        pos = wrap_idx
                        dir = edge_rot[current_edge]
                    else:
                        break
            else:
                break

            positions.append(tuple(pos))
            assert map[tuple(pos)] == 0


print(sum(pos * [1_000, 4]) + dir)

# %%

# part1: 43466
# part2: 162155

# %%


draw_map = map.copy()
tail = 100


def update(i):
    global draw_map, tail
    draw_map[positions[i]] = 2
    if i >= tail:
        if positions[i - tail] not in positions[i - tail + 1:i + 1]:
            draw_map[positions[i - tail]] = 0
    ax.clear()
    plt.imshow(draw_map)


# len(positions)
fig, ax = plt.subplots(figsize=(12, 12))
ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=5)
# ani.save('animation.gif', writer='imagemagick')
ani.save("animation.mp4")
plt.show()

# %%
# %%
# %%
# %%
