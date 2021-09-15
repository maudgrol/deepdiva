from utils.transformer import preset_to_patch
import spiegelib as spgl

def split_train_override_patch(patch, train_parameter_list):
    '''This function takes a patch and a list of parameter indices and
    returns 2 list of tuples: first a list containing only those tuples, that are in the train_parameter_list,
    and second a list of tuples with the remaining parameter

    it should be used when there is a patch, and you quickly want a sublist of that patch with only the trainable parameter tuples
    (and/or a sublist with only the override parameter tuples)'''


    #i use a set so that it is robust to a list that contains the same parameter twice (it will only erase it once)
    plist = set(train_parameter_list)
    patch_copy = list(patch)
    #i sort the list so that it start removing from left to right, otherwise the indexing would be wrong
    for i, tuples in enumerate(sorted(plist)):
        patch.remove(patch[tuples-i])
    train_parameter = patch

    override_parameter = list(set(patch_copy)-set(train_parameter))
    return override_parameter, train_parameter


# i want to use this patch for basic training
base_patch = preset_to_patch("../data/MS-REV1_deepdiva.h2p")

#i only want to vary 4 parameters, so all other 278 should be overridden
## i removed 86 tuning oscillator 1, 244 wave oscillator 1, 140 & 148 cutoff, 141& 149 resonance"
train_parameters, override_parameters = split_train_override_patch(base_patch, [86, 87, 90, 97, 98, 131, 132, 140, 141, 148, 149])


synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
                            note_length_secs=1.0,
                            render_length_secs=3.0,
                            overridden_params= override_parameters)

# Mel-frequency Cepstral Coefficients audio feature extractor.
features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                              hop_size=1024, time_major=True)

# Setup generator for MFCC output

for i in range(10):
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder="../ms20_data_FM_mfcc", save_audio=False)
    generator.generate(5000, file_prefix=f"train_{i}_")
    generator.generate(1000, file_prefix=f"test_{i}_")
    generator.save_scaler('data_scaler.pkl')







