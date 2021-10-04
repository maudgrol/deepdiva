from utils.patch_utils import preset_to_patch, split_train_override_patch
import spiegelib as spgl

PATH = "../small_data6"
BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"
PARAMETERS_TO_OVERRIDE = [(0,0.5), #main volume is fixed to 50%
                          (3,0.5), #led colour .. 0 effect on sound
                          (16,0.0), #not priority -- has no effect if a single note is played, set to "last"
                          (43, 0), (54, 0), (143, 0), (154, 0), #these are all keyfollow parameters.. they make no sense if we show the model only one note
                          (168, 0.5), #AMP volume is also fixed to 50%
                          (167, 0.5), (172, 0), (173, 0.5 ), #these things affect the panorama.. since we are MONO this makes no sense at all ... we could include them if we opt for STEREO sound...
                          (176, 0.5 ), (177, 0.5 ), #these is just the visual scope... have nothing to do with sound at all
                          (175, 0.5) #i actually dont know what this knob does but i think its more neutral at 0.5 than 0
                          ]

FX_PARAMETERS = [(1, 0), (2, 0), ]

# setting up the synth, plays a certain note (c2) with a certain velocity
synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
                            note_length_secs=3.0,
                            render_length_secs=6.0,
                            overridden_params=PARAMETERS_TO_OVERRIDE)

# Mel-frequency Cepstral Coefficients audio feature extractor.
features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                              hop_size=1024, time_major=True)

# Setup generator for MFCC output

for i in range(1):
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder=PATH, save_audio=False, scale=True)
    generator.generate(10_000, file_prefix=f"train_{i}_")
    generator.generate(1000, file_prefix=f"test_{i}_")

    generator.save_scaler('data_scaler.pkl')
