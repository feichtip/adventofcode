import pandas as pd
import numpy as np


df = pd.read_csv('input', names=['calories'], skip_blank_lines=False)
np.sum(np.sort([np.nansum(cal) for cal in np.split(df.calories.values, np.where(df.isna())[0])])[::-1][:3])
