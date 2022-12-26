import re
import string

import numpy as np
import pandas as pd

# %%

df = pd.read_csv('input', header=None)
trees = df[0].values
trees = np.asarray([[int(tree) for tree in row] for row in trees])

# %%

total_trees = trees.shape[0] * trees.shape[1]

visible_trees = np.full_like(trees, 0, dtype=int)
reverse_slice = slice(None, None, -1)
nom_slice = slice(None)
for trans, slice_j in [([0, 1], nom_slice), ([0, 1], reverse_slice), ([1, 0], nom_slice), ([1, 0], reverse_slice)]:
    visible = [[(i, j) for j, tree in enumerate(row) if (tree > np.asarray(row[:j])).all()]
               for i, row in enumerate(trees.transpose(trans)[:, slice_j])]
    visible = [idx for row in visible for idx in row]
    for idx in visible:
        visible_trees.transpose(trans)[:, slice_j][idx] += 1


# %%

scenic_score = np.full_like(trees, 1, dtype=int)
for i, row in enumerate(trees):
    for j, tree in enumerate(row):
        for k, tall_trees in enumerate([np.where(trees[i, j + 1:] >= tree)[0],
                                        np.where((trees[i, :j] >= tree)[::-1])[0],
                                        np.where((trees[:i, j] >= tree)[::-1])[0],
                                        np.where(trees[i + 1:, j] >= tree)[0]]):
            if tall_trees.size == 0:
                if k == 0:
                    scenic_score[i, j] *= trees.shape[1] - j - 1
                elif k == 1:
                    scenic_score[i, j] *= j
                elif k == 2:
                    scenic_score[i, j] *= i
                elif k == 3:
                    scenic_score[i, j] *= trees.shape[0] - i - 1
            else:
                scenic_score[i, j] *= np.min(tall_trees) + 1

scenic_score.max()
# %%
