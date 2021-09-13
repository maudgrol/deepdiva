import os
import numpy as np
import pandas as pd
import pickle
import json
import spiegelib as spgl

from norm_functions import *
from scale_functions import *
from scaling import patch_transform

DATA_PATH = "../data/"
VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"

# Load function dictionaries
with open(os.path.join(DATA_PATH, "normalize_dict.pickle"), 'rb') as handle:
    norm_dict = pickle.load(handle)

with open(os.path.join(DATA_PATH, "scaling_dict.pickle"), 'rb') as handle:
    scale_dict = pickle.load(handle)

# Load h2p dictionary template
with open(os.path.join(DATA_PATH, "h2p_template.json")) as f:
  parameter_dictionary = json.load(f)

header_dictionary = {
    "Bank": "Diva Factory Best Of",
    "Author": "Bronto Scorpio",
    "Description": "push the modulation wheel",
    "Usage": "MW = filter\\r\\nAT = vibrato"
}


def spiegelib_array_to_h2p(header_dictionary, parameter_dictionary, filename):
    with open(os.path.join(DATA_PATH, f"{filename}.h2p"), "w") as f:

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

        # write parameters
        for index in parameter_dictionary.keys():
            label = parameter_dictionary[index]["label"]
            value = parameter_dictionary[index]["value"]
            f.write(f"{label}={value}")
            f.write("\n")

spiegelib_array_to_h2p(header_dictionary, parameter_dictionary, "h2p_example")


ROW_TO_PARAMETER_INDEX ={'32': 0,
                         '33': 1,
                         '34': 2,
                         '42': 3,
                         '47': 256,
                         '48': 257,
                         '49': 258,
                         '52': 259,
                         '53': 260,
                         '54': 261,
                         '55': 262,
                         '56': 263,
                         '57': 271,
                         '59': 4,
                         '60': 5,
                         '61': 6,
                         '62': 7,
                         '63': 8,
                         '64': 9,
                         '65': 10,
                         '66': 11,
                         '67': 12,
                         '69': 13,
                         '72': 14,
                         '73': 15,
                         '82': 16,
                         '109': 20,
                         '112': 21,
                         '114': 22,
                         '116': 23,
                         '118': 24,
                         '119': 25,
                         '120': 26,
                         '121': 27,
                         '122': 28,
                         '123': 29,
                         '124': 30,
                         '125': 31,
                         '126': 32,
                         '129': 33,
                         '130': 34,
                         '131': 35,
                         '132': 36,
                         '133': 37,
                         '134': 38,
                         '135': 39,
                         '136': 40,
                         '137': 41,
                         '138': 42,
                         '139': 43,
                         '141': 44,
                         '142': 45,
                         '143': 46,
                         '144': 47,
                         '145': 48,
                         '146': 49,
                         '147': 50,
                         '148': 51,
                         '149': 52,
                         '150': 53,
                         '151': 54,
                         '153': 55,
                         '154': 56,
                         '155': 57,
                         '156': 58,
                         '157': 272,
                         '158': 59,
                         '159': 60,
                         '160': 61,
                         '161': 62,
                         '162': 63,
                         '163': 64,
                         '165': 65,
                         '166': 66,
                         '167': 67,
                         '168': 68,
                         '169': 273,
                         '170': 69,
                         '171': 70,
                         '172': 71,
                         '173': 72,
                         '174': 73,
                         '175': 74,
                         '177': 75,
                         '178': 76,
                         '179': 77,
                         '180': 78,
                         '181': 79,
                         '182': 80,
                         '183': 81,
                         '184': 82,
                         '185': 83,
                         '186': 84,
                         '188': 85,
                         '189': 86,
                         '190': 87,
                         '191': 88,
                         '192': 89,
                         '193': 90,
                         '194': 91,
                         '195': 92,
                         '196': 93,
                         '197': 94,
                         '198': 95,
                         '199': 96,
                         '200': 97,
                         '201': 98,
                         '202': 99,
                         '203': 100,
                         '204': 101,
                         '205': 102,
                         '206': 103,
                         '207': 104,
                         '208': 105,
                         '209': 106,
                         '210': 107,
                         '211': 108,
                         '212': 109,
                         '213': 110,
                         '214': 111,
                         '215': 112,
                         '216': 113,
                         '217': 114,
                         '218': 115,
                         '219': 116,
                         '220': 117,
                         '221': 118,
                         '222': 119,
                         '223': 120,
                         '224': 121,
                         '225': 122,
                         '226': 123,
                         '227': 124,
                         '228': 125,
                         '229': 126,
                         '230': 127,
                         '231': 128,
                         '232': 129,
                         '233': 130,
                         '234': 131,
                         '235': 132,
                         '236': 133,
                         '237': 134,
                         '238': 135,
                         '239': 136,
                         '240': 137,
                         '241': 138,
                         '248': 278,
                         '249': 264,
                         '250': 279,
                         '251': 265,
                         '252': 266,
                         '253': 280,
                         '255': 139,
                         '256': 140,
                         '257': 141,
                         '258': 142,
                         '259': 143,
                         '260': 144,
                         '261': 145,
                         '262': 146,
                         '264': 147,
                         '265': 148,
                         '266': 149,
                         '267': 150,
                         '268': 151,
                         '269': 152,
                         '270': 153,
                         '271': 154,
                         '272': 155,
                         '273': 156,
                         '274': 157,
                         '275': 158,
                         '276': 159,
                         '277': 160,
                         '278': 161,
                         '279': 162,
                         '280': 163,
                         '281': 164,
                         '282': 165,
                         '283': 166,
                         '285': 267,
                         '286': 268,
                         '287': 269,
                         '288': 270,
                         '290': 167,
                         '291': 168,
                         '292': 169,
                         '293': 170,
                         '294': 171,
                         '295': 172,
                         '296': 173,
                         '297': 174,
                         '298': 175,
                         '301': 176,
                         '302': 177,
                         '309': 178,
                         '311': 179,
                         '312': 180,
                         '313': 181,
                         '314': 182,
                         '316': 183,
                         '317': 184,
                         '318': 185,
                         '319': 186,
                         '320': 187,
                         '321': 188,
                         '322': 189,
                         '323': 274,
                         '324': 275,
                         '326': 190,
                         '327': 191,
                         '328': 192,
                         '329': 193,
                         '330': 194,
                         '331': 195,
                         '332': 196,
                         '334': 197,
                         '335': 198,
                         '336': 199,
                         '337': 200,
                         '338': 201,
                         '339': 202,
                         '340': 203,
                         '341': 204,
                         '342': 205,
                         '343': 206,
                         '345': 207,
                         '346': 208,
                         '347': 209,
                         '348': 210,
                         '349': 211,
                         '350': 212,
                         '351': 213,
                         '352': 214,
                         '353': 215,
                         '354': 216,
                         '356': 217,
                         '358': 218,
                         '359': 219,
                         '360': 220,
                         '361': 221,
                         '363': 222,
                         '364': 223,
                         '365': 224,
                         '366': 225,
                         '367': 226,
                         '368': 227,
                         '369': 228,
                         '370': 276,
                         '371': 277,
                         '373': 229,
                         '374': 230,
                         '375': 231,
                         '376': 232,
                         '377': 233,
                         '378': 234,
                         '379': 235,
                         '381': 236,
                         '382': 237,
                         '383': 238,
                         '384': 239,
                         '385': 240,
                         '386': 241,
                         '387': 242,
                         '388': 243,
                         '389': 244,
                         '390': 245,
                         '392': 246,
                         '393': 247,
                         '394': 248,
                         '395': 249,
                         '396': 250,
                         '397': 251,
                         '398': 252,
                         '399': 253,
                         '400': 254,
                         '401': 255}

def h2p_file_to_h2p_parameters_dictionary(file):
    def clean_string(string):
        try:
            value = float(key_value[1])

        except:
            value = key_value[1].strip("'")
        return value

    # Get parameters in a list
    h2p_parameters_list = file.readlines()

    end_of_header_index = h2p_parameters_list.index("*/\n") + 2
    end_of_parameters_index = end_of_header_index + 402

    h2p_parameters_list = h2p_parameters_list[end_of_header_index:end_of_parameters_index]

    # get parameters in a dictionary
    h2p_parameter_values = []
    for line in h2p_parameters_list:
        key_value = line.strip().split("=")
        value = clean_string(key_value[1])

        h2p_parameter_values.append(value)

    return h2p_parameter_values


with open(os.path.join(DATA_PATH, "h2p_example.h2p")) as file:
    # put h2p parameters into a list of tuples
    h2p_row_values = h2p_file_to_h2p_parameters_dictionary(file)
    # remove string parameters
    h2p_row_values[h2p_row_values.index('Chorus1')] = 0.0
    h2p_row_values[h2p_row_values.index('Plate2')] = 0.0
    h2p_row_values[h2p_row_values.index('Delay2')] = 0.0

    # extract parameters that spiegelib uses
    row_indicies = [int(x) for x in ROW_TO_PARAMETER_INDEX.keys()]
    parameter_indicies = ROW_TO_PARAMETER_INDEX.values()

    h2p_row_values = np.asarray(h2p_row_values)

    # keep only values that
    h2p_row_values = list(h2p_row_values[row_indicies])

    h2p_row_parameter_index = []
    for parameter, row_value in zip(parameter_indicies, h2p_row_values):
        h2p_row_parameter_index.append([parameter, float(row_value)])

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

    #     non_normal = [(x[0], x[1]) for x in sorted_h2p_row_parameter_index]

normalized_patch


synth = spgl.synth.SynthVST(VST_PATH,
                            note_length_secs=1.0,
                            render_length_secs=4.0)

synth.set_patch(normalized_patch)
synth.render_patch()
audio = synth.get_audio()
audio.save(os.path.join(DATA_PATH, "audio_test.wav"), normalize=False)





# Set synthesizer parameters and render audio file
#synth.set_patch()
#synth.render_patch()
#synth.get_audio()