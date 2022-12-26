import re
import string

import numpy as np
import pandas as pd

df = pd.read_csv('input', header=None)
m = df.loc[0, 0]

# re.search(f'', m)

n_distinct = 4
[i for i in range(len(m)) if len(set(m[i:i + n_distinct])) == n_distinct][0] + n_distinct

n_distinct = 14
[i for i in range(len(m)) if len(set(m[i:i + n_distinct])) == n_distinct][0] + n_distinct

# %%
