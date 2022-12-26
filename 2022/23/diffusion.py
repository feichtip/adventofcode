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

part2 = True

if part2:
    rounds = 5_000
    edge_size = 40
    shift = -20
else:
    rounds = 10
    edge_size = 10
    shift = 0

# file = np.loadtxt('test_input', dtype=object, delimiter='\n', comments=None)
file = np.loadtxt('input', dtype=object, delimiter='\n', comments=None)

map_size = (len(file) + 2 * edge_size, len(file[0]) + 2 * edge_size)


# initialise
map = np.full(map_size, 0, dtype=int)
for i, row in enumerate(file):
    map[i + edge_size + shift, edge_size + shift:-edge_size + shift] = [0 if sign == '.' else (1 if sign == '#' else -1) for sign in row]

plt.figure(figsize=(8, 8))
plt.imshow(map)
plt.show()

n_elfs = map.sum()
elfs = {elf: (i, j) for elf, (i, j) in enumerate(zip(*np.where(map)))}
next_dir = {elf: 0 for elf in range(n_elfs)}
check_dirs = [np.array([[-1, i] for i in [-1, 0, 1]]),  # N
              np.array([[+1, i] for i in [-1, 0, 1]]),  # S
              np.array([[i, -1] for i in [-1, 0, 1]]),  # W
              np.array([[i, +1] for i in [-1, 0, 1]]),  # E
              ]

movements = []
for round in range(rounds):
    # propose positions
    proposals = {elf: None for elf in range(n_elfs)}
    for elf, pos in elfs.items():
        # print(elf, pos + check_dirs[next_dir[elf]])
        if round > 0:
            next_dir[elf] = (next_dir[elf] + 1) % 4

        neighbours = [map[tuple(p)] for dir in range(4) for p in (pos + check_dirs[dir])]
        if sum(neighbours) == 0:
            # print(elf, map[pos])
            continue

        for dir in [i % 4 for i in range(next_dir[elf], next_dir[elf] + 4)]:
            check = check_dirs[dir]
            occupation = [map[tuple(p)] for p in (pos + check)]
            # print(elf, dir, sum(occupation))
            if sum(occupation) == 0:
                proposal = pos + check[1]
                proposals[elf] = tuple(proposal)
                break

    u, i, c = np.unique([val[0] * 1_000 + val[1] for val in proposals.values() if val], return_counts=True, return_index=True)
    # elf can move to valid positions
    valid = np.array([val for val in proposals.values() if (val is not None)])[i[c == 1]]

    # print(round, valid)
    # move to valid position
    moving_elfs = [elf for elf, pos in proposals.items() if (pos is not None) and (pos == valid).all(axis=1).any()]

    if len(moving_elfs) == 0:
        print(f'no more elfs are moving in round {round + 1}')
        plt.figure(figsize=(8, 8))
        plt.imshow(map)
        plt.show()
        break

    moving_proposals = {elf: proposals[elf] for elf in moving_elfs}
    movements.append(moving_proposals.copy())
    for elf, proposed_pos in moving_proposals.items():
        map[elfs[elf]] = 0
        elfs[elf] = proposed_pos
        map[proposed_pos] = 1

    if part2 and not round % 100 == 0:
        continue

    plt.figure(figsize=(8, 8))
    plt.imshow(map)
    plt.show()

if not part2:
    i, j = np.where(map)
    map_zoom = map[i.min():i.max() + 1, j.min():j.max() + 1]
    plt.figure(figsize=(8, 8))
    plt.imshow(map_zoom)
    plt.show()

    print((map_zoom == 0).sum())

# part1: 4172
# part2: 942

# %%
len(movements) + 1
# %%

map = np.full(map_size, 0, dtype=int)
for i, row in enumerate(file):
    map[i + edge_size + shift, edge_size + shift:-edge_size + shift] = [0 if sign == '.' else (1 if sign == '#' else -1) for sign in row]
elfs = {elf: (i, j) for elf, (i, j) in enumerate(zip(*np.where(map)))}


def update(i):
    global map, movements
    for elf, proposed_pos in movements[i].items():
        map[elfs[elf]] = 0
        elfs[elf] = proposed_pos
        map[proposed_pos] = 1
    ax.clear()
    plt.imshow(map)


# len(positions)
fig, ax = plt.subplots(figsize=(14, 14))
ani = animation.FuncAnimation(fig, update, frames=len(movements), interval=20)
# ani.save('animation.gif', writer='imagemagick')
ani.save("animation.mp4")
plt.show()
