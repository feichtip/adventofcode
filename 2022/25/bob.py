import copy
import random
import re
import string
from ast import literal_eval
from pprint import pprint

import awkward as ak
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from sympy import Eq, solve, symbols
from tqdm import tqdm

# %%


# file = np.loadtxt('test_input', dtype=object, delimiter='\n', comments=None)
file = np.loadtxt('input', dtype=object, delimiter='\n', comments=None)

numbers = [[-2 if c == '=' else (-1 if c == '-' else int(c)) for c in line] for line in file]

# %%


def from_snafu(x):
    n = 0
    for p, d in enumerate(x[::-1]):
        n += d * 5**p
    return n


def to_snafu(n):
    d_map = {0: 0, 1: 1, 2: 2, 3: -2, 4: -1}
    x = []
    p = 0
    while n != from_snafu(x):
        d = d_map[(n - from_snafu(x)) % 5**(p + 1) // 5**p]
        x.insert(0, d)
        p += 1
    return x


def format_snafu(x):
    x = ['=' if d == -2 else ('-' if d == -1 else str(d)) for d in x]
    return ''.join(x)


fuel = sum([from_snafu(x) for x in numbers])
print('part 1:')
print(format_snafu(to_snafu(fuel)))

# %%
