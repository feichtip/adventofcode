import copy
import random
import re
import string
from ast import literal_eval
from pprint import pprint

import awkward as ak
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from sympy import Eq, solve, symbols
from tqdm import tqdm

# %%


# file = np.loadtxt('test_input', dtype=object, delimiter='\n', comments=None)
file = np.loadtxt('input', dtype=object, delimiter='\n', comments=None)

map_size = (len(file), len(file[0]))

# %%

directions = {'>': 1, 'v': 2, '<': 3, '^': 4}
add_i = {1: 0, 2: 1, 3: 0, 4: -1}
add_j = {1: 1, 2: 0, 3: -1, 4: 0}

# initialise
init_map = np.full(map_size, 0, dtype=int)
draw_map = np.full(map_size, 0, dtype=int)
for i, row in enumerate(file):
    init_map[i, :] = [0 if sign == '.' else (-1 if sign == '#' else directions[sign]) for sign in row]
    draw_map[i, :] = [-1 if sign == '#' else 0 for sign in row]


init_blizzards = {}
for dir in range(1, 5):
    init_blizzards[dir] = np.where(init_map == dir)


def create_maps(blizzards, final_t=300):
    global draw_map
    blizzards = copy.deepcopy(blizzards)
    maps = []
    blizz = []
    for t in range(final_t + 1):
        map = draw_map != -1
        for dir in range(1, 5):
            map[blizzards[dir]] = False
        maps.append(map.copy())
        blizz.append(copy.deepcopy(blizzards))
        if t == 0:
            plt.figure(figsize=(8, 8))
            plt.imshow(map)
            plt.show()

        for dir in range(1, 5):
            i, j = blizzards[dir]
            i += add_i[dir]
            j += add_j[dir]
            i[i == map_size[0] - 1] = 1
            j[j == map_size[1] - 1] = 1
            i[i == 0] = map_size[0] - 2
            j[j == 0] = map_size[1] - 2
            blizzards[dir] = (i, j)
    return maps, blizz


def create_graph(maps, start_weight=1, target_weight=0):
    G = nx.DiGraph()
    for t, (map, prev_map) in enumerate(zip(maps[1:], maps[:-1])):
        for i, j in zip(*np.where(prev_map)):
            if i == 0:  # start
                weight = start_weight
                deltas = [[0, 0], [1, 0]]
            elif i == map_size[0] - 1:  # finish
                weight = target_weight
                deltas = [[0, 0], [-1, 0]]
            else:
                weight = 1
                deltas = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]

            for di, dj in deltas:
                if map[i + di, j + dj]:
                    G.add_edge((t, i, j), (t + 1, i + di, j + dj), weight=weight)
    return G


def find_path(blizzards, direction, final_t=300):
    """
    direction 0: start->finish
    direction 1: finish->start
    """
    global map_size

    maps, blizz = create_maps(blizzards, final_t=final_t)
    G = create_graph(maps, start_weight=abs(1 - direction), target_weight=direction)

    if direction == 0:
        source = (0, 0, 1)
        target = (final_t, map_size[0] - 1, map_size[1] - 2)
    elif direction == 1:
        source = (0, map_size[0] - 1, map_size[1] - 2)
        target = (final_t, 0, 1)

    path = nx.shortest_path(G, source=source, target=target, weight='weight')
    at_target = [p for p in path if p[1] == target[1] and p[2] == target[2]]
    reached_target = at_target[0][0]
    print(f'reached target in minute {reached_target}')
    return reached_target, blizz[reached_target]

# %%


final_t = 300

reached_target_a, blizzards_a = find_path(init_blizzards, direction=0, final_t=final_t)
reached_target_b, blizzards_b = find_path(blizzards_a, direction=1, final_t=final_t)
reached_target_c, blizzards_c = find_path(blizzards_b, direction=0, final_t=final_t)

print(f'part1: {reached_target_a}')
print(f'part2: {reached_target_a + reached_target_b + reached_target_c}')

# %%

# # show graph
# plt.figure(figsize=(12, 12))
# pos = nx.spring_layout(G)
# pos = {}
# for node in G.nodes:
#     all_t = [n for n in G.nodes if n[0] == node[0]]
#     pos[node] = [node[0], all_t.index(node)]
# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edges(G, pos, arrows=False)
# plt.show()
