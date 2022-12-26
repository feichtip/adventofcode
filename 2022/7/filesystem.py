import re
import string

import numpy as np
import pandas as pd

# %%

df = pd.read_csv('input', header=None)
cmds = df[0].values

cd_indices = np.array([i for i, c in enumerate(cmds) if c[:4] == '$ cd'])
paths = ['/'.join([cmd[5:] for cmd in cmds[cd_indices[:i + 1]]])[1:] for i, cd_idx in enumerate(cd_indices)]

while np.any(['/..' in path for path in paths]):
    for i, path in enumerate(paths):
        spans = [match.span() for match in re.finditer('(/\w+/\.\.)', path)][::-1]
        for span in spans:
            paths[i] = paths[i][:span[0]] + paths[i][span[1]:]

filesizes = [[paths[np.where(cd_indices < i)[0][-1]], int(c.split(' ')[0])] for i, c in enumerate(cmds) if (c[0] != '$') and (c[:3] != 'dir')]

dirsizes = {path: 0 for path in paths}
for path, size in filesizes:
    dirsizes[path] += size

full_dirsizes = {path: 0 for path in paths}
for path in dirsizes.keys():
    full_dirsizes[path] = sum([dirsize for dirpath, dirsize in dirsizes.items() if path in dirpath])

# part 1
sum([val for val in full_dirsizes.values() if val <= 100000])

unused = 70000000 - full_dirsizes['']
required = 30000000

# part 2
min([val for val in full_dirsizes.values() if val >= required - unused])
