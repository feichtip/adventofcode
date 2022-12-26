import re
import string

import numpy as np
import pandas as pd

# %%

df = pd.read_csv('input', header=None, sep=' ')
dir_map = {'R': (0, 1), 'L': (0, -1), 'U': (1, 1), 'D': (1, -1)}

# %%

size = 1_000
rope_len = 10

grid = np.zeros((size, size), dtype=int)
rope_pos = np.array([[size / 2, size / 2]] * rope_len, dtype=int)

grid[rope_pos[rope_len - 1][0], rope_pos[rope_len - 1][1]] += 1

for dir, steps in df.values:
    cord, move = dir_map[dir]
    for _ in range(steps):
        rope_pos[0][cord] += move  # head moves

        for i in range(rope_len - 1):
            offset = rope_pos[i] - rope_pos[i + 1]
            if (np.abs(offset) > 1).any():
                # make a move
                to_move = np.abs(offset) >= 1
                rope_pos[i + 1][to_move] += np.sign(offset)[to_move]
                if i == (rope_len - 2):
                    grid[rope_pos[rope_len - 1][0], rope_pos[rope_len - 1][1]] += 1

(grid > 0).sum()
