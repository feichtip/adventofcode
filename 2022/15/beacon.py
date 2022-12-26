import copy
import re
import string
from pprint import pprint

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

# %%

test = False

if test:
    scaling = 1
    sensors_text = np.loadtxt('test_input', dtype=object, delimiter='\n')
    min_x = -5
    max_x = 25
    n_points = 1_000_000
else:
    scaling = 1
    sensors_text = np.loadtxt('input', dtype=object, delimiter='\n')
    min_x = -2_000_000
    max_x = 5_000_000
    n_points = 100_000


sensors = np.array([[int(pos) for pos in re.findall(r'(-*\d+)', sensor)] for sensor in sensors_text])
s_x, s_y, b_x, b_y = sensors.T / scaling


# %%


def add_points(arr, n=2):
    return np.stack([np.append(arr[:-1] + np.diff(arr) * i / n, arr[-1]) for i in range(n)]).T.flatten()

# %%


part2 = False

if part2:
    plot = False
    iter_range = range(3_186_980, 3_186_990)
else:
    plot = True
    if test:
        iter_range = [10]
    else:
        iter_range = [2_000_000]

check_overlap = np.linspace(min_x / scaling, max_x / scaling, max_x - min_x + 1)

for check_y in tqdm(iter_range):
    overlap = np.full_like(check_overlap, False, dtype=bool)
    radii = abs(b_x - s_x) + abs(b_y - s_y) + 0.5
    intervals = []

    if plot:
        figure, axes = plt.subplots(1, figsize=(18, 18))
        plt.hlines(check_y, check_overlap.min(), check_overlap.max(), color='k', lw=1)

    for i, (x, y, radius) in enumerate(zip(s_x, s_y, radii)):
        a = np.concatenate([np.linspace(x - radius, x, int(n_points / 2)), np.linspace(x, x + radius, int(n_points / 2))])
        a = np.concatenate([a, a[::-1]])
        b = np.concatenate([np.linspace(y, y + radius, int(n_points / 2)), np.linspace(y + radius, y, int(n_points / 2))])
        b = np.concatenate([b, y + y - b])

        if plot:
            axes.plot(a, b)

        close_to_line = np.isclose(b, check_y, rtol=1E-2)
        if close_to_line.sum() > 0:
            if not test:
                y_strip = np.abs(b - check_y) < 50
                a = add_points(a[y_strip], 100)
                b = add_points(b[y_strip], 100)

            close_to_line_idx = np.argsort(np.abs(b - check_y))[:4]
            a_close = a[close_to_line_idx]
            b_close = b[close_to_line_idx]
            min_overlap = a_close.argmin()
            max_overlap = a_close.argmax()
            intervals.append([a_close[min_overlap], a_close[max_overlap]])
            # print(i, a_close[min_overlap], a_close[max_overlap])
            if plot:
                plt.plot([a_close[min_overlap], a_close[max_overlap]], [b_close[min_overlap], b_close[max_overlap]], marker='', ls='-', color='red', lw=3)
            overlap = ((check_overlap > a_close[min_overlap]) & (check_overlap < a_close[max_overlap])) | overlap

    search_region = (check_overlap >= 0) & (check_overlap <= 4_000_000)
    if ((~overlap[search_region]).sum() > 0):
        gap_x = np.where(~overlap[search_region])[0].item()
        gap_y = check_y
        tuning_frequency = gap_x * 4_000_000 + gap_y
        print(f'found gap at ({gap_x}, {gap_y})')
        print(f'tuning frequency: {tuning_frequency}')

    if plot:
        plt.plot(b_x, b_y, ls='', marker='o', color='k')
        # axes.set_aspect(1)
        if not test:
            plt.hlines(4000000, 0, 4000000)
            plt.hlines(0, 0, 4000000)
            plt.vlines(0, 0, 4000000)
            plt.vlines(4000000, 0, 4000000)

        # plt.vlines(3_335_000, 0, 4000000)
        # plt.hlines(3_186_900, 0, 4000000)
        # plt.ylim(3_186_980, 3_186_990)
        # plt.xlim(3_334_450, 3_334_500)
        plt.show()

    if not part2:
        for beacon in list(set(b_x[b_y == check_y])):
            overlap[int(beacon)] = False
        print(overlap.sum())
# %%

# 5256611
# 13337919186981
