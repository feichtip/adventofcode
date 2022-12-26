import copy
import random
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

part2 = True

# file = np.loadtxt('test_input', dtype=int, delimiter='\n')
file = np.loadtxt('input', dtype=int, delimiter='\n')

if part2:
    key = 811589153
    rounds = 10
else:
    key = 1
    rounds = 1

file = [(i, num * key) for i, num in enumerate(file)]
original_file = copy.deepcopy(file)

for _ in range(rounds):
    for item in original_file:
        # print([f[1] for f in file])
        current_pos = file.index(item)
        target_pos = (current_pos + item[1]) % (len(file) - 1)
        # print(item[1], current_pos, target_pos)
        file.pop(current_pos)
        if (target_pos == 0) and (target_pos < current_pos):
            file.insert(len(file), item)
        else:
            file.insert(target_pos, item)

final_list = [f[1] for f in file]


pos = []
for delta in [1000, 2000, 3000]:
    pos.append(final_list[(final_list.index(0) + delta) % len(file)])

sum(pos)


# part1: 13522
# part2: 17113168880158

# %%
