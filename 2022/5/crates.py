import string

import numpy as np
import pandas as pd

df = pd.read_csv('input', skipfooter=505, header=None, engine='python')
boxes = [[s[i + 1:i + 2] for i in range(0, len(s), 4)] for s in df[0].values]
n_stacks = len(boxes[0])
stacks = [[row[n_box] for row in boxes[::-1] if row[n_box] != ' '] for n_box in range(n_stacks)]

df_instructions = pd.read_csv('input', skiprows=9, delimiter=' ', header=None)
for instruction in df_instructions[[1, 3, 5]].values:
    n, src, dst = instruction

    # CrateMover 9000
    # for i in range(n):
    #     box = stacks[src - 1].pop()
    #     stacks[dst - 1].append(box)

    # CrateMover 9001
    boxes = stacks[src - 1][-n:]
    stacks[src - 1] = stacks[src - 1][:-n]
    stacks[dst - 1] = stacks[dst - 1] + boxes

# %%

last_boxes = [stack[-1] for stack in stacks]
print(''.join(last_boxes))

# %%
