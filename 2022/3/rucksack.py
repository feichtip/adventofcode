import pandas as pd
import numpy as np
import string

df = pd.read_csv('input', names=['contents'])
compartments = np.array([[str[:int(len(str) / 2)], str[int(len(str) / 2):]] for str in df.contents.values])


rep = [[s for s in compartment[0] if s in compartment[1]] for compartment in compartments]

string.ascii_uppercase

val_map = {letter: i + 1 for i, letter in enumerate(string.ascii_lowercase)}
val_map.update({letter: i + 1 + len(string.ascii_lowercase) for i, letter in enumerate(string.ascii_uppercase)})

sum([val_map[r[0]] for r in rep])

# %%

rucksacks = np.split(df.contents.values, np.arange(3, len(df), 3))

rep = [[s for s in rucksack[0] if (s in rucksack[1]) and (s in rucksack[2])] for rucksack in rucksacks]
sum([val_map[r[0]] for r in rep])
