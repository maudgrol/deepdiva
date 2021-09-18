#!/usr/bin/env python
import copy
import numpy as np
import pandas as pd

from deepdiva.utils.dictionaries import *

class H2P():

    def __init__(self):
        """
        Constructor
        """

        self.parameter_dict = get_row_to_parameter_dict()
        self.norm_dict = get_norm_dict()
        self.scale_dict = get_scale_dict()
        self.parameter_index_to_h2p_tag_list = get_parameter_index_to_h2p_tag_list()
        self.default_header = {
            "Bank": "Diva Factory Best Of",
            "Author": "Bronto Scorpio",
            "Description": "push the modulation wheel",
            "Usage": "MW = filter\\r\\nAT = vibrato"}


    def preset_to_patch(self, h2p_filename:str, normalize=True):
        """
        Transforms a h2p preset file to a patch with parameter values, optionally normalizing values.
        :param h2p_filename: Name (string) of the h2p preset file
        :param normal: Boolean to normalize patch values, defaults to true
        :return: List with (normalized) patch values
        """

        with open(h2p_filename) as file:
            h2p = self.__h2p_raw_to_h2p_param(file)

        # Certain strings have to be transformed into numbers
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

        # Extract relevant parameters
        row_indicies = [int(x) for x in self.parameter_dict.keys()]
        parameter_indicies = self.parameter_dict.values()
        h2p = np.asarray(h2p)
        h2p = list(h2p[row_indicies])

        h2p_row_parameter_index = []
        for parameter, row_value in zip(parameter_indicies, h2p):
            h2p_row_parameter_index.append([parameter, float(row_value)])

        # Add parameters that are not in H2P preset file, but are expected in patch
        h2p_row_parameter_index.append([17, 1.0])
        h2p_row_parameter_index.append([18, 1.0])
        h2p_row_parameter_index.append([19, 1.0])

        # Order h2p_row_parameter_index
        sorted_h2p_row_parameter_index = sorted(h2p_row_parameter_index, key=lambda x: x[0])

        patch_values = [x[1] for x in sorted_h2p_row_parameter_index]
        normalized_patch_values = self.patch_transform(self.norm_dict, patch_values)

        # replace values with normalized values
        normalized_patch = [(x[0], normalized_patch_values[i]) for i, x in enumerate(sorted_h2p_row_parameter_index)]

        non_normalized_patch = [(x[0], x[1]) for x in sorted_h2p_row_parameter_index]

        if normalize:
            return normalized_patch
        else:
            return non_normalized_patch


    def patch_to_preset(self, patch, h2p_filename:str):
        """

        :param patch:
        :param h2p_filename:
        :return:
        """

        patch_values = [x[1] for x in patch]
        normalized_patch_values = self.patch_transform(self.scale_dict, patch_values)

        new_patch = [(x[0], normalized_patch_values[i]) for i, x in enumerate(patch)]
        patch_dict = dict(new_patch)

        with open(h2p_filename, "w") as f:
            # write header
            f.write("/*@Meta")
            f.write("\n\n")
            for header in self.default_header.keys():
                f.write(f"{header}:")
                f.write("\n")
                f.write(f"'{self.default_header[header]}'")
                f.write("\n\n")
            f.write("*/")
            f.write("\n\n")

            # Update the parameter dictionary values
            h2p_parameter_list = copy.deepcopy(self.parameter_index_to_h2p_tag_list)

            for index, h2p_parameter in enumerate(h2p_parameter_list):
                if h2p_parameter["parameter_index"] != None:
                    patch_value = patch_dict[h2p_parameter["parameter_index"]]
                    h2p_parameter_list[index]['value'] = patch_value

            # Write parameters
            for entry in h2p_parameter_list:
                label = entry["label"]
                value = entry["value"]
                f.write(f"{label}={value}")
                f.write("\n")


    def patch_transform(self, function_dict:dict, patch):
        """
        Transforms the values in a patch according to the functions defined in the function dictionary
        :param function_dict: dictionary with relevant transformation functions
        :param patch: list of current patch values
        :return: transformed patch numpy array
        """

        patch_df = pd.DataFrame(patch).transpose()
        transformed_patch = patch_df.agg(function_dict)
        transformed_patch = transformed_patch.T.to_numpy().flatten()

        return transformed_patch


    def _h2p_raw_to_h2p_param(self, file):
        """
        Helper function to extract parameters from h2p preset file
        :param file: inherets file from h2p_raw_to_h2p_param function
        :return: list of h2p preset parameters
        """
        def clean_string(string):
            try:
                value = float(key_value[1])
            except:
                value = key_value[1].strip("'")
            return value

        # Extract text from h2p preset file
        h2p = file.readlines()

        # Remove header and binary data at end of h2p preset file
        end_of_header_index = h2p.index("*/\n") + 2
        end_of_parameters_index = h2p.index("#cm=Rtary2\n") + 11
        h2p_body = h2p[end_of_header_index:end_of_parameters_index]

        # Add h2p preset parameters to a list and remove irrelvant parameters
        h2p_parameters = []
        for line in h2p_body:
            key_value = line.strip().split("=")

            # These parameters have no equivalent in the patch
            parameters_to_skip = ["PSong", "rMW", "rPW"]
            if key_value[0] in parameters_to_skip:
                continue

            value = clean_string(key_value[1])

            h2p_parameters.append(value)

        return h2p_parameters
