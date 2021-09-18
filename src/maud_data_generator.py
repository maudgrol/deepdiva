#!/usr/bin/env python
from utils.patch_utils import preset_to_patch, split_train_override_patch
import spiegelib as spgl
import os

DATA_PATH = "../data"
VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"

BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"
PARAMETERS_TO_RANDOMIZE = [86, 131, 148, 149]

# Use this patch for model training
base_patch = preset_to_patch(BASE_PATCH, normal=True)

# Vary 4 parameters, override 278 parameters
# Trainable parameters: 86 tuning oscillator 1, 244 wave oscillator 1, 148 cutoff, 149 resonance"
train_parameters, override_parameters = split_train_override_patch(base_patch, PARAMETERS_TO_RANDOMIZE)

# Set up the synth
synth = spgl.synth.SynthVST(VST_PATH,
                            note_length_secs=1.0,
                            render_length_secs=1.0,
                            overridden_params= override_parameters)

# Set up Mel-frequency Cepstral Coefficients audio feature extractor
features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                              hop_size=1024, time_major=True)

# Setup data generator for MFCC output with audio saved for mel spectrogram
for i in range(1):
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder=os.path.join(DATA_PATH, "toy_data"),
                                      save_audio=True,
                                      scale=True)
    generator.generate(10000, file_prefix=f"train_{i}_")
    generator.generate(1000, file_prefix=f"test_{i}_")
    generator.save_scaler('data_scaler.pkl')


# Setup data generator for evaluation data audio files
generator_eval = spgl.DatasetGenerator(synth, features,
                                       output_folder=os.path.join(DATA_PATH, "toy_eval_data"),
                                       save_audio=True)
generator_eval.generate(10, file_prefix="eval_")