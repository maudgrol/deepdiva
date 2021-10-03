import copy
import numpy as np
import pandas as pd

from utils.norm_functions import *
from utils.scale_functions import *
from utils.dictionaries import *

#DATA_PATH = "../../data/"

# dictionaries
norm_dict = get_norm_dict()
scale_dict = get_scale_dict()
parameter_index_to_h2p_tag_list = get_parameter_index_to_h2p_tag_list()

DEFAULT_HEADER = {
    "Bank": "Diva Factory Best Of",
    "Author": "Bronto Scorpio",
    "Description": "push the modulation wheel",
    "Usage": "MW = filter\\r\\nAT = vibrato"
}

def patch_transform(function_dict:dict, patch):
    patch_df = pd.DataFrame(patch).transpose()
    transformed_patch = patch_df.agg(function_dict)
    transformed_patch = transformed_patch.transpose().to_numpy().flatten()

    return transformed_patch

def preset_to_patch(h2p_filename, normal=True):
    # Load function dictionaries
    norm_dict = get_norm_dict()
    scale_dict = get_scale_dict()
    parameter_dict = get_row_to_parameter_dict()


    # function that checks if there is a string in there
    def h2p_raw_to_h2p_param(file):
        def clean_string(string):
            try:
                value = float(key_value[1])
            except:
                value = key_value[1].strip("'")
            return value

        # Get parameters in a list
        h2p = file.readlines()

        # get rid of the header and the binary stuff in the end
        end_of_header_index = h2p.index("*/\n") + 2
        end_of_parameters_index = end_of_header_index + 402

        h2p_body = h2p[end_of_header_index:end_of_parameters_index]

        # get parameters in a dictionary
        h2p_param = []
        for line in h2p_body:
            key_value = line.strip().split("=")
            value = clean_string(key_value[1])

            h2p_param.append(value)

        return h2p_param

    with open(h2p_filename) as file:
        # put h2p parameters into a list of tuples
        h2p = h2p_raw_to_h2p_param(file)

        # these strings have to be put into numbers by hand
        def fx_numeric(fx, value):
            try:
                h2p[h2p.index(fx)] = value
            except:
                pass

        fx_numeric("Chorus1", 0.0)
        fx_numeric("Phaser1", 1.0)
        fx_numeric("Plate1", 2.0)
        fx_numeric("Delay1", 3.0)
        fx_numeric("Rotary1", 4.0)
        fx_numeric("Chorus2", 0.0)
        fx_numeric("Phaser2", 1.0)
        fx_numeric("Plate2", 2.0)
        fx_numeric("Delay2", 3.0)
        fx_numeric("Rotary2", 4.0)

        # extract parameters that spiegelib uses
        row_indicies = [int(x) for x in parameter_dict.keys()]
        parameter_indicies = parameter_dict.values()

        h2p = np.asarray(h2p)

        # keep only values that
        h2p = list(h2p[row_indicies])

        h2p_row_parameter_index = []
        for parameter, row_value in zip(parameter_indicies, h2p):
            h2p_row_parameter_index.append([parameter, float(row_value)])

        # add three parameters that are not in H2P, but expected in spiegelib
        h2p_row_parameter_index.append([17, 1.0])
        h2p_row_parameter_index.append([18, 1.0])
        h2p_row_parameter_index.append([19, 1.0])

        # order h2p_row_parameter_index
        sorted_h2p_row_parameter_index = sorted(h2p_row_parameter_index, key=lambda x: x[0])

        # mauds function
        patch_values = [x[1] for x in sorted_h2p_row_parameter_index]
        normalized_patch_values = patch_transform(norm_dict, patch_values)

        # replace values with normalized values
        normalized_patch = [(x[0], normalized_patch_values[i]) for i, x in enumerate(sorted_h2p_row_parameter_index)]

        non_normal = [(x[0], x[1]) for x in sorted_h2p_row_parameter_index]

        if normal == 1:
            return normalized_patch
        elif normal == 0:
            return non_normal

def patch_to_preset(patch, filename, header_dictionary=DEFAULT_HEADER):

    ######### MIGHT NEED TO LOOK INTO THIS CODE FOR ORDERING ISSUES
    patch_values = [x[1] for x in patch]
    normalized_patch_values = patch_transform(scale_dict, patch_values)

    new_patch = [(x[0], normalized_patch_values[i]) for i, x in enumerate(patch)]
    patch_dict = dict(new_patch)
    #########

    with open(filename, "w") as f:

        # write header
        f.write("/*@Meta")
        f.write("\n\n")
        for header in header_dictionary.keys():
            f.write(f"{header}:")
            f.write("\n")
            f.write(f"'{header_dictionary[header]}'")
            f.write("\n\n")
        f.write("*/")
        f.write("\n\n")

        # update the parameter dictionary values
        h2p_parameter_list = copy.deepcopy(parameter_index_to_h2p_tag_list)
        for index, h2p_parameter in enumerate(h2p_parameter_list):
            if h2p_parameter["parameter_index"] != None:
                patch_value = patch_dict[h2p_parameter["parameter_index"]]
                h2p_parameter_list[index]['value'] = patch_value

        # write parameters
        for entry in h2p_parameter_list:
            label = entry["label"]
            value = entry["value"]
            f.write(f"{label}={value}")
            f.write("\n")


def split_train_override_patch(patch, train_parameter_list:list):
    """
    This function returns a list of tuples containing only those parameters that are in the train_parameter_list,
    and a second list of tuples with the overridden parameters
    :param patch: original patch - list of tuples
    :param train_parameter_list: list of trainable parameters (parameter IDs)
    :return: Two lists of tuples: overridden parameters, trainable parameters
    """

    # List of unique trainable parameters
    param_list = set(train_parameter_list)

    # Create deep copies from original patch as to not edit original patch
    temp_trainable = copy.deepcopy(patch)
    temp_overridden = copy.deepcopy(patch)

    # Remove trainable parameters from list of overridden parameters (from last to first)
    for element in sorted(param_list, reverse=True):
        temp_overridden.pop(element)

    override_parameter_tuples = temp_overridden
    train_parameter_tuples = list(set(temp_trainable)-set(override_parameter_tuples))
    return override_parameter_tuples, train_parameter_tuples

def get_randomization_small():
    """
    this function  provides a list with parameter names to randomize
    this small sized set contains basic knobs of the ms20
    :return: a list of integers ( parameter numbers)
    """
    random_parameters = [
        33, 34, 35, #Attack, Decay, Sustain of he ENV1
        86, 87, 89, 90, 91, 92, 97, 98, 131, 132, #oscillator section MS-20
        140, 141, 148, 149, 155 #2 Filters ms-20
    ]
    return random_parameters

def get_randomization_medium():
    """
    this function  provides a list with parameter names to randomize
    this medium sized set contains basically all the knobs of the ms20 (substracting the modular part of it)
    also, release parameters are not included because we do not record any release audio
    :return: a list of integers ( parameter numbers)
    """
    random_parameters = [
        33, 34, 35, #Attack, Decay, Sustain of he ENV1
        44, 45, 46, 47, #Attack, Decay, Sustain of he ENV2
        55, 57, 62, #sync, rate and waveform LFO 1
        65, 67, 72, # sync, rate and waveform LFO 2
        86, 87, 89, 90, 91, 92, 97, 98, 131, 132, #oscillator section MS-20
        104, 145, 151, #Env2 modulating Tune1, HPF and VCF1
        106, 153, #LFO2 modulating Tune1 and VCF1
        140, 141, 148, 149, 155 #2 Filters ms-20
    ]
    return random_parameters

def get_randomization_big():
    """
    this function  provides a list with parameter names to randomize
    this big  set contains 124 parameters of the DIVA
    it excludes everything about releases, key follows, the LFO's, the Effects Section
    :return: a list of integers ( parameter numbers)
    """
    random_parameters = []
    random_parameters.extend(range(4, 16))
    random_parameters.extend([33, 34, 35])
    random_parameters.extend(range(37, 43))
    random_parameters.extend([44, 45, 46])
    random_parameters.extend(range(48, 54))
    random_parameters.extend(range(85, 143))
    random_parameters.extend(range(144, 154))
    random_parameters.extend(range(155, 167))
    random_parameters.extend([169, 170, 171, 174])
    random_parameters.extend(range(264, 271))
    random_parameters.extend([278, 279, 280])
    return random_parameters
