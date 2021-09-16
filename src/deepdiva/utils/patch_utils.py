import numpy as np
import pandas as pd


def split_train_override_patch(patch, train_parameter_list):
    """
    This function returns a list of tuples containing only those parameters that are in the train_parameter_list,
    and a second list of tuples with the overridden parameters
    :param patch:
    :param train_parameter_list: list of trainable parameters (parameter IDs)
    :return: Two lists of tuples: overridden parameters, trainable parameters
    """

    # List of unique trainable parameters
    param_list = set(train_parameter_list)

    # Create list from patch
    patch_copy = list(patch)

    #i sort the list so that it start removing from left to right, otherwise the indexing would be wrong
    for i, tuples in enumerate(sorted(param_list)):
        patch.remove(patch[tuples-i])

    train_parameter_tuples = patch
    override_parameter_tuples = list(set(patch_copy)-set(train_parameter_tuples))
    return override_parameter_tuples, train_parameter_tuples