import re
import string

import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm

# %%

# n_rounds = 20
# div_by = 3

n_rounds = 10_000
div_by = 1


monkeys = pd.read_csv('input', header=None, sep='\n').values.reshape(-1, 6)

item_sets = [[int(i) for i in re.findall('(\d+)', monkey)] for monkey in monkeys[:, 1]]
operations = [eval('lambda old: ' + re.search('new = (.+)', monkey).group(1)) for monkey in monkeys[:, 2]]
test_divs = [int(re.search("divisible by (.+)", monkey).group(1)) for monkey in monkeys[:, 3]]
tests = [eval(f'lambda x: x % {test_div} == 0') for test_div in test_divs]
if_true = [int(re.search('(\d+)', monkey).group(1)) for monkey in monkeys[:, 4]]
if_false = [int(re.search('(\d+)', monkey).group(1)) for monkey in monkeys[:, 5]]

mod_val = np.prod(test_divs)

inspections = [0] * len(item_sets)
for round in tqdm(range(n_rounds)):
    for monkey, (operation, items, test, t, f) in enumerate(zip(operations, item_sets, tests, if_true, if_false)):
        inspections[monkey] += len(items)

        worry_levels = [(operation(item) // div_by) % mod_val for item in items]
        [item_sets[t if test(worry_level) else f].append(worry_level) for worry_level in worry_levels]  # throw stuff
        item_sets[monkey] = []

        # [item_sets[t if test(operation(item) // div_by) else f].append(operation(items.pop(0)) // div_by) for item in items.copy()]  # throw stuff

print(f'monkey business: {np.product(sorted(inspections)[-2:])}')

# 1: 58794
# 2: 20151213744
