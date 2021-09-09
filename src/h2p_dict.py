import os
import numpy as np
import json
import pandas as pd

DATA_PATH = "../data/"

df = pd.read_csv(os.path.join(DATA_PATH, "mapping parameters 2.csv"), sep=";", header=None)

starting_row = 10
h2p_dict = {}
h2p_index = 0
for i in np.arange(starting_row, df.shape[0]):
    h2p_dict[h2p_index] = {"parameter_index": df.loc[i, 3],
                           "value": df.loc[i, 2],
                           "label": df.loc[i, 1]}
    h2p_index += 1

# Need to actually keep nan to separate real 0 from missing 0??
#dict_clean = {k: {k2: 0 if pd.isna(v2) else v2 for k2, v2 in v.items()} for k, v in h2p_dict.items()}
