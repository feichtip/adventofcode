import copy
import re
import string
from pprint import pprint

import awkward as ak
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

# %%

# droplet = np.loadtxt('test_input', delimiter=',').astype(int)
droplet = np.loadtxt('input', delimiter=',').astype(int)

dim_size = droplet.max() + 1
space = np.zeros((dim_size,) * 3)
holes = np.zeros((dim_size,) * 3)
for drop in droplet:
    space[tuple(drop)] = 1

edges = 0
for a_scan in range(dim_size):
    plt.imshow(space[a_scan, :, :])
    plt.show()
    for b_scan in range(dim_size):
        slices = [(a_scan, slice(None), b_scan),
                  (slice(None), a_scan, b_scan),
                  (a_scan, b_scan, slice(None)),
                  ]
        for sl in slices:
            if space[sl].sum() > 0:
                lava_i = np.where(space[sl])[0]
                # print(np.diff(lava_i))
                start = lava_i[:-1][np.diff(lava_i) > 1] + 1
                end = lava_i[1:][np.diff(lava_i) > 1]
                for aa, bb in zip(start, end):
                    tup = tuple([s if s != slice(None) else slice(aa, bb) for s in sl])
                    # print(tup)
                    holes[tup] += 1
                edges += 2 + 2 * (np.diff(lava_i) > 1).sum()


print(edges)
# part1: 4400

# %%

for i in range(dim_size):
    for j in range(dim_size):
        for k in range(dim_size):
            if holes[i, j, k] != 3:
                holes[i, j, k] = 0
            if (np.array([i, j, k]) == 0).any() or (np.array([i, j, k]) == (dim_size - 1)).any():
                holes[i, j, k] = 0
                continue
            neighbours = [(i + 1, j, k),
                          (i - 1, j, k),
                          (i, j + 1, k),
                          (i, j - 1, k),
                          (i, j, k + 1),
                          (i, j, k - 1)]
            for neighbour in neighbours:
                if (holes[neighbour] != 3) and (space[neighbour] == 0):
                    holes[i, j, k] = 0

hole_edges = 0
for a_scan in range(dim_size):
    for b_scan in range(dim_size):
        slices = [(a_scan, slice(None), b_scan),
                  (slice(None), a_scan, b_scan),
                  (a_scan, b_scan, slice(None)),
                  ]
        for sl in slices:
            if holes[sl].sum() > 0:
                holes_i = np.where(holes[sl] == 3)[0]
                hole_edges += 2 + 2 * (np.diff(holes_i) > 1).sum()

print(hole_edges)

# %%

print(edges - hole_edges)
# part2: 2522
