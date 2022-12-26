import re
import string

import numpy as np
import pandas as pd
from pprint import pprint

# %%

cums = np.cumsum(np.array([val for val in pd.read_csv('../10/input', header=None)[0].map(lambda cmd: (1, 0) if cmd == 'noop' else (2, int(cmd[5:])))]), axis=0)
insatances = [1] + [1 + cums[cums[:, 0] < iter][-1][1] for iter in range(2, cums[:, 0][-1] + 1)]


print(np.sum([inst * iter for iter, inst in enumerate(insatances, start=1) if iter in [20, 60, 100, 140, 180, 220]]))

chars = ['#' if abs((pos % 40) - inst) <= 1 else '.' for pos, inst in enumerate(insatances, start=0)]
pprint([''.join(chars[i * 40:(i + 1) * 40]) for i in range(6)])
