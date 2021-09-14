import os
import numpy as np
import json
import pandas as pd

DATA_PATH = "../data/"
STARTING_ROW = 10
H2P_INDEX = 0

df = pd.read_csv(os.path.join(DATA_PATH, "mapping_parameters_5.csv"), sep=";", header=0)

def build_h2p_template(starting_row: int, h2p_index: int, df: pd.DataFrame):
    '''
    :param starting_row: First row to read data from in dataframe
    :param h2p_index: Starting row number for h2p template
    :param df: Dataframe that includes relevant information to create build initial dictionary.
               Needs to contain following columns: h2plabel, value, paramid
    :return: dictionary that maps h2p-file row to synthbase parameter id, value and h2p-label
    '''
    h2p_dict = {}

    for i in np.arange(starting_row, df.shape[0]):
        h2p_dict[h2p_index] = {"parameter_index": df.loc[i, 'param_id'],
                               "value": df.loc[i, 'h2p_value'],
                               "label": df.loc[i, 'h2p_label']}

        h2p_index += 1

        h2p_dict_clean = {k: {k2: None if pd.isna(v2) else v2 for k2, v2 in v.items()} for k, v in h2p_dict.items()}

    return h2p_dict_clean

# Build initial dictionary from csv file
h2p_template = build_h2p_template(STARTING_ROW, H2P_INDEX, df)

# Save initial dictionary
with open(os.path.join(DATA_PATH, "h2p_template.json"), 'w') as fp:
    json.dump(h2p_template, fp)

# Load h2p dictionary template
with open(os.path.join(DATA_PATH, "h2p_template.json")) as f:
  h2p_template = json.load(f)


# Load test patch for update function
testParams = np.load(os.path.join(DATA_PATH, "data_mfcc/test_patches.npy"))
patch = testParams[0]


def update_h2p_template(h2p_template: dict, synth_patch):
    '''
    :param h2p_dict: Initial h2p template for which values will be updated
    :param synth_patch: Patch vector containing the values to be used
    :return: Updated dictionary from which h2p file can be created
    '''
    for k, v in h2p_template.items():
        for i in np.arange(len(synth_patch)):
            if v['parameter_index'] == i:
                v['value'] = synth_patch[i]

    return h2p_template

updated_template = update_h2p_template(h2p_template, patch)


