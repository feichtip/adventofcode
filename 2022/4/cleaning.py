import pandas as pd
import numpy as np
import string

df = pd.read_csv('input', names=['a', 'b'])

splitter = lambda x: [int(s) for s in x.split('-')]

contained = 0
for a_range, b_range in zip(df.a.map(splitter), df.b.map(splitter)):
    # if (a_range[0] == b_range[0]) and (a_range[1] == b_range[1]):
    if a_range == b_range:
        contained += 1
        print('equal')
        continue
    for range1, range2 in [[a_range, b_range], [b_range, a_range]]:
        if (range1[0] <= range2[0]):
            if (range2[1] <= range1[1]):
                contained += 1

contained
# %%
len(df)

overlap = 0
for a_range, b_range in zip(df.a.map(splitter), df.b.map(splitter)):
    if (a_range[1] >= b_range[0]) and (b_range[1] >= a_range[0]):
        overlap += 1
overlap
