import copy
import re
import string
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%

elevation = np.array([list(line) for line in pd.read_csv('input', header=None)[0].values])

start_pos = tuple([i.item() for i in np.where(elevation == 'S')])
end_pos = tuple([i.item() for i in np.where(elevation == 'E')])
elevation[start_pos] = 'a'
elevation[end_pos] = 'z'

vfunc = np.vectorize(string.ascii_lowercase.index)
elevation = vfunc(elevation)

# %%

# add ring of zeros
elev_mat = np.full((elevation.shape[0] + 2, elevation.shape[1] + 2), -1, dtype=float)
elev_mat[1:-1, 1:-1] = elevation
start_pos = tuple([i + 1 for i in start_pos])
end_pos = tuple([i + 1 for i in end_pos])


# %%


import heapq
import sys


class Graph:

    def __init__(self):
        self.vertices = {}

    def add_vertex(self, name, edges):
        self.vertices[name] = edges

    def shortest_path(self, start, finish):
        distances = {}  # Distance from start to node
        previous = {}  # Previous node in optimal path from source
        nodes = []  # Priority queue of all nodes in Graph

        for vertex in self.vertices:
            if vertex == start:  # Set root node as distance of 0
                distances[vertex] = 0
                heapq.heappush(nodes, [0, vertex])
            else:
                distances[vertex] = sys.maxsize
                heapq.heappush(nodes, [sys.maxsize, vertex])
            previous[vertex] = None

        while nodes:
            smallest = heapq.heappop(nodes)[1]  # Vertex in nodes with smallest distance in distances
            if smallest == finish:  # If the closest node is our target we're done so print the path
                path = []
                while previous[smallest]:  # Traverse through nodes til we reach the root which is 0
                    path.append(smallest)
                    smallest = previous[smallest]
                return path
            if distances[smallest] == sys.maxsize:  # All remaining vertices are inaccessible from source
                # print('inaccessible')
                break

            for neighbor in self.vertices[smallest]:  # Look at all the nodes that this vertex is attached to
                alt = distances[smallest] + self.vertices[smallest][neighbor]  # Alternative path distance
                if alt < distances[neighbor]:  # If there is a new shortest path update our priority queue (relax)
                    distances[neighbor] = alt
                    previous[neighbor] = smallest
                    for n in nodes:
                        if n[1] == neighbor:
                            n[0] = alt
                            break
                    heapq.heapify(nodes)
        return distances

    def __str__(self):
        return str(self.vertices)


# %%

for scenic in [False, True]:  # True for part 2
    g = Graph()  # create graph
    for i in range(1, elev_mat.shape[0] - 1):
        for j in range(1, elev_mat.shape[1] - 1):
            neighbors = [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]
            neighbor_dict = {neig: 1 for neig in neighbors if ((elev_mat[neig] - elev_mat[i, j]) <= 1) and (elev_mat[neig] != -1)}
            if ((i, j) == start_pos) and scenic:
                neighbor_dict.update({(ii, jj): 0 for ii, jj in zip(*np.where(elev_mat == 0))})
            g.add_vertex((i, j),  neighbor_dict)

            # g.add_vertex((i, j),  {neig: 1 for neig in neighbors if ((elev_mat[neig] - elev_mat[i, j]) <= 1) and (elev_mat[neig] != -1)})

    shortest = g.shortest_path(start_pos, end_pos)

    if scenic:
        # minus one because of old start tile
        print(f'shortest path length: {len(shortest) - 1}')
    else:
        print(f'shortest path length: {len(shortest)}')

    elev_mat_img = elev_mat.copy()
    plt.figure(figsize=(14, 14))
    plt.plot(np.array(shortest).T[1], np.array(shortest).T[0], color='orange', lw=1)
    plt.plot(start_pos[1], start_pos[0], marker='x', color='red', ms='5')
    plt.plot(end_pos[1], end_pos[0], marker='x', color='red', ms=6)
    plt.imshow(elev_mat_img, origin='upper')
    plt.show()

# %%
