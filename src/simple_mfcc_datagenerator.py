from utils.patch_utils import preset_to_patch, split_train_override_patch
import spiegelib as spgl

PATH = "../small_data5predict"
BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"
PARAMETERS_TO_RANDOMIZE = [33, 34, 35, 36, 86, 131, 148, 149]

# i want to use this patch for basic training
base_patch = preset_to_patch(BASE_PATCH)

# i only want to vary some parameters, so all other 278 should be overridden
train_parameters, override_parameters = split_train_override_patch(base_patch, PARAMETERS_TO_RANDOMIZE)

# setting up the synth and feature class , i basically copied the values in the DEXED paper
synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
                            note_length_secs=3.0,
                            render_length_secs=6.0,
                            overridden_params=override_parameters)

# Mel-frequency Cepstral Coefficients audio feature extractor.
features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                              hop_size=1024, time_major=True)

# Setup generator for MFCC output

for i in range(1):
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder=PATH, save_audio=True, scale=True)
    generator.generate(20, file_prefix=f"train_{i}_")
    generator.generate(1, file_prefix=f"test_{i}_")

    generator.save_scaler('data_scaler.pkl')
