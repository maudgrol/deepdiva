import os
import librenderman as rm
import numpy as np
from utils.h2p_utils import H2P
from utils.patch_utils import split_train_override_patch, get_randomization_medium, get_randomization_tiny4
import librosa

DATA_PATH = "../data/"
FOLDER_NAME = "data_tiny4"
RANDOM_PARAMETERS = sorted(get_randomization_tiny4())  #overridden
SAMPLE_SIZE = 1000
TOTAL_BATCHES = 1000

engine = rm.RenderEngine(44100, 512, 512)
engine.load_plugin("/Library/Audio/Plug-Ins/VST/u-he/Diva.vst")

generator = rm.PatchGenerator(engine)

h2p = H2P()
base_patch = h2p.preset_to_patch(h2p_filename="../data/MS-REV1_deepdiva.h2p")

# Create one data example to infer size of arrays
example_patch = generator.get_random_patch()
engine.set_patch(example_patch)
engine.render_patch(48, 127, 2.0, 2.0)
example_audio = engine.get_audio_frames()
example_audio = np.asarray(example_audio)
mfcc_example = librosa.feature.mfcc(
    y=example_audio,
    n_fft=2048,
    win_length=2048,
    hop_length=1024,
    n_mfcc=13,
    sr=44100,
    fmin=50,
    fmax=15000
)

# get overidden parameters
overridden_parameters, _ = split_train_override_patch(base_patch, RANDOM_PARAMETERS)
overridden = [p[0] for p in overridden_parameters]

# Initiate empty arrays to save data samples
mfcc_set = np.zeros((SAMPLE_SIZE, mfcc_example.shape[0], mfcc_example.shape[1]), dtype=np.float32)
patch_set = np.zeros((SAMPLE_SIZE, len(RANDOM_PARAMETERS)), dtype=np.float32)

def make_a_batch(batch_number):
    try:
        for i in range(SAMPLE_SIZE):
            # Generate random patch
            random_patch = generator.get_random_patch()
            engine.set_patch(random_patch)

            # Overwrite fixed parameters and get full patch
            for param in overridden_parameters:
                engine.override_plugin_parameter(param[0], param[1])
            patch = engine.get_patch()

            # Render the patch
            engine.render_patch(48, 127, 2.0, 2.0)

            # Get the MFCC
            audio = engine.get_audio_frames()
            audio = np.asarray(audio)
            mfcc = librosa.feature.mfcc(
                y=audio,
                n_fft=2048,
                win_length=2048,
                hop_length=1024,
                n_mfcc=13,
                sr=44100,
                fmin=50,
                fmax=15000
            )
            print(mfcc.shape)

            #audio = np.array(audio, copy=True, dtype=np.float32)
            mfcc_set[i] = mfcc

            # Get synthesizer patch
            for element in sorted(overridden, reverse=True):
                patch.pop(element)
            patch_set[i] = [p[1] for p in patch]

        np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"{batch_number}_mfcc.npy"), mfcc_set)
        np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"{batch_number}_patches.npy"), patch_set)
        batch_number += 1

        if batch_number > TOTAL_BATCHES:
            return
        else:
            make_a_batch(batch_number)

    except:
        if batch_number <= TOTAL_BATCHES:
            print("RESTARTING!")
            make_a_batch(batch_number)


make_a_batch(82)