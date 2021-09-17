from utils.patch_utils import preset_to_patch, split_train_override_patch
import spiegelib as spgl

PATH = "../small_data4"
BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"
PARAMETERS_TO_RANDOMIZE = [86, 131, 148, 149]

# i want to use this patch for basic training
base_patch = preset_to_patch(BASE_PATCH)

#i only want to vary 4 parameters, so all other 278 should be overridden
## i removed 86 tuning oscillator 1, 244 wave oscillator 1, 140 & 148 cutoff, 141& 149 resonance"
train_parameters, override_parameters = split_train_override_patch(base_patch, PARAMETERS_TO_RANDOMIZE)

#setting up the synth and feature class , i basically copied the values in the DEXED paper
synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
                            note_length_secs=1.0,
                            render_length_secs=1.0,
                            overridden_params= override_parameters)

# Mel-frequency Cepstral Coefficients audio feature extractor.
features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                              hop_size=1024, time_major=True)

# Setup generator for MFCC output
for i in range(1):
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder=PATH, save_audio=False, scale=True)
    generator.generate(10000, file_prefix=f"train_{i}_")
    generator.generate(1000, file_prefix=f"test_{i}_")
    generator.save_scaler('data_scaler.pkl')







