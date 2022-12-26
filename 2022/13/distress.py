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

signal = pd.read_csv('input', header=None, sep=r'\n', engine='python')[0].values
# signal = pd.read_csv('test_input', header=None, sep=r'\n', engine='python')[0].values

lefts = signal[::2]
rights = signal[1::2]


def check_order(left, right):
    if isinstance(left, list) and isinstance(right, list):
        for l, r in zip(left, right):
            # print(l, r)
            return_val = check_order(l, r)
            if return_val is not None:
                return return_val
        if len(left) < len(right):
            return True
        elif len(right) < len(left):
            return False
    elif isinstance(left, int) and isinstance(right, int):
        if left < right:
            return True
        elif right < left:
            return False
    else:
        if isinstance(left, int):
            left = [left]
        elif isinstance(right, int):
            right = [right]
        return_val = check_order(left, right)
        if return_val is not None:
            return return_val

# %%


# part 1
correct_idx = [i for i, (left, right) in enumerate(zip(lefts, rights), start=1) if check_order(literal_eval(left), literal_eval(right))]
print(np.sum(correct_idx))

# %%

# part 2


def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        # if order is correct
        if check_order(array[j], pivot):
            i = i + 1
            (array[i], array[j]) = (array[j], array[i])

    (array[i + 1], array[high]) = (array[high], array[i + 1])
    return i + 1


def quickSort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        quickSort(array, low, pi - 1)
        quickSort(array, pi + 1, high)


packets = [literal_eval(left) for left in lefts] + [literal_eval(right) for right in rights] + [[[2]], [[6]]]
size = len(packets)
quickSort(packets, 0, size - 1)
print((packets.index([[2]]) + 1) * (packets.index([[6]]) + 1))

# %%
# %%
# %%
# %%

# solution with numpy arrays, does not work


def get_shape(arr, lengths={}, indices=()):
    if isinstance(arr, list):
        if len(arr) == 0:
            lengths[indices] = -1
        else:
            for i, arr_i in enumerate(arr):
                get_shape(arr_i, lengths, indices + (i,))
    else:
        lengths[indices] = arr
    return lengths


n_levels = 0
size = 0
for idx, (left, right) in enumerate(zip(lefts, rights), start=1):
    for shape in [get_shape(eval(left)), get_shape(eval(right))]:
        max_levels = max([len(key) for key in shape.keys()])
        max_size = max([max(key) for key in shape.keys() if len(key) > 0]) + 1
        n_levels = max_levels if (max_levels > n_levels) else n_levels
        size = max_size if (max_size > size) else size

print(n_levels, size)

# %%


def fill_array(arr, full_arr=np.full((size,) * n_levels, -1000, dtype=int), indices=(), level=n_levels + 2):
    if isinstance(arr, list):
        if len(arr) == 0:
            full_arr[indices + (0,) * (len(full_arr.shape) - len(indices))] = -level
        for i, arr_i in enumerate(arr):
            full_arr = fill_array(arr_i, full_arr.copy(), indices + (i,), level - 1)
    else:
        full_arr[indices + (0,) * (len(full_arr.shape) - len(indices))] = arr
    return full_arr.copy()


correct_idx = []
for idx, (left, right) in enumerate(zip(lefts, rights), start=1):
    full_left = fill_array(eval(left))
    full_right = fill_array(eval(right))
    smaller = full_left <= full_right
    # print(idx, len(eval(left)), len(eval(right)), correct_order)
    # print(idx)

    if (full_right[~smaller] == -1000).all():
        # edge case, don't need to compare lenght of lists

        larger_idx = np.array(np.where(~smaller)).T

        # print([(full_left[tuple(l_idx)], full_right[tuple(l_idx)]) for l_idx in larger_idx])

        # left_first_entry = np.full_like(larger_idx, False, dtype=bool)
        # right_first_entry = np.full_like(larger_idx, False, dtype=bool)
        # for row, add_j in zip(*np.where(larger_idx == 0)):
        #     test_idx = larger_idx[row].copy()
        #     test_idx[add_j] += 1
        #     # entries even though index == 0
        #     left_first_entry[row, add_j] = full_left[tuple(test_idx)] != -1000
        #     right_first_entry[row, add_j] = full_right[tuple(test_idx)] != -1000
        #
        # both_first = left_first_entry & right_first_entry
        # different_level = np.any(~both_first & (right_first_entry | left_first_entry), axis=1)
        # print(larger_idx[different_level])

        # if not different_level.any():

        # print([(full_left[tuple(l_idx)], full_right[tuple(l_idx)]) for l_idx in larger_idx])

        # decrease last non-zero by one
        larger_idx[range(len(larger_idx)), [np.where(l_idx)[0][-1] for l_idx in larger_idx]] -= 1

        # print(larger_idx)
        # print([(full_left[tuple(l_idx)], full_right[tuple(l_idx)]) for l_idx in larger_idx])

        exceptions = [((full_left[tuple(l_idx)] < full_right[tuple(l_idx)]) and full_left[tuple(l_idx)] >= 0) or (full_right[tuple(l_idx)] == -1000) for l_idx in larger_idx]
        veto = np.all(exceptions) and (len(exceptions) > 0)

        # else:
        #     veto = False
    else:
        veto = False

    if smaller.all() or veto:
        correct_idx.append(idx)
    # if idx == 84:
    #     break


print(correct_idx)
# 431, 670, 2261, 2350, 2750 false
np.sum(correct_idx)


# %%
