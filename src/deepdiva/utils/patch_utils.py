#!/usr/bin/env python
import copy


def split_train_override_patch(patch, train_parameter_list:list):
    """
    This function returns a list of tuples containing only those parameters that are in the train_parameter_list,
    and a second list of tuples with the overridden parameters
    :param patch:
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
        del temp_overridden[element]

    override_parameter_tuples = temp_overridden
    train_parameter_tuples = list(set(temp_trainable)-set(override_parameter_tuples))
    return override_parameter_tuples, train_parameter_tuples