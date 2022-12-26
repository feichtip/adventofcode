import copy
import random
import re
import string
from ast import literal_eval
from pprint import pprint

import awkward as ak
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from sympy import Eq, solve, symbols
from tqdm import tqdm

# %%

part2 = True

# file = np.loadtxt('test_input', dtype=object, delimiter='\n')
file = np.loadtxt('input', dtype=object, delimiter='\n')

number_monkeys = {line[:4]: int(line[6:]) for line in file if len(line) < 11}
op_monkeys = {line[:4]: re.findall(r'\w+', line[6:]) + [line[11]] for line in file if len(line) >= 11}

if part2:
    number_monkeys['humn'] = 'x'


def calculate(monkey, expression=''):
    if monkey in op_monkeys:
        mon1, mon2, op = op_monkeys[monkey]
        to_eval = f'({calculate(mon1, expression)}) {op} ({calculate(mon2, expression)})'
        # print(to_eval)
        return expression + to_eval
    else:
        return number_monkeys[monkey]


if not part2:
    res = calculate('root')
    print(int(eval(res)))
    # part1: 49288254556480
else:
    x = symbols('x')
    eq1 = Eq(eval(calculate(op_monkeys['root'][0])), eval(calculate(op_monkeys['root'][1])))
    sol = solve(eq1)
    print(int(sol[0]))
    # part2: 3558714869436
