#!/usr/bin/env python
from utils.patch_utils import preset_to_patch, split_train_override_patch
import spiegelib as spgl
import os
import numpy as np
import tensorflow as tf
import soundfile

DATA_PATH = "../data"
VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
FOLDER_NAME = "dataset_test"

# Generated samples per data batch
train_size = 10000
test_size = 1000
eval_size = 10
# Number of data batches
nr_batches = 1

BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"
PARAMETERS_TO_RANDOMIZE = [86, 131, 148, 149]

# i want to use this patch for basic training
base_patch = preset_to_patch(BASE_PATCH)

#i only want to vary 4 parameters, so all other 278 should be overridden
## i removed 86 tuning oscillator 1, 244 wave oscillator 1, 140 & 148 cutoff, 141& 149 resonance"
train_parameters, override_parameters = split_train_override_patch(base_patch, PARAMETERS_TO_RANDOMIZE)


overridden_parameters = [(0, 0.5), # main volume is fixed to 50%
                          (3, 0.5), # led colour - no effect on sound
                          (16,0.0), # note priority - has no effect if a single note is played, set to "last"
                          (43, 0), # these are all keyfollow parameters - they make no sense if we show the model only one note
                          (54, 0),
                          (143, 0),
                          (154, 0),
                          (168, 0.5), # AMP volume is also fixed to 50%
                          (167, 0.5),
                          (172, 0),
                          (177, 0.5), # visual scope - no effect on sound
                          (175, 0.5)] # not sure what this knob does but neutral at 0.5


def wav_to_binary(file):
    # Loading and decoding the wav file.
    audio_binary = tf.io.read_file(file)
    try:
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)
    except:
      data, samplerate = soundfile.read(file)
      soundfile.write(file, data, samplerate, subtype='PCM_16')
      audio_binary = tf.io.read_file(file)
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)

    return audio[:, 0]


# Synthesizer configuration
synth = spgl.synth.SynthVST(VST_PATH,
                            note_length_secs=2.0,
                            render_length_secs=2.0,
                            overridden_params=override_parameters)

# Set up Mel-frequency Cepstral Coefficients audio feature extractor
features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                              hop_size=1024, time_major=True)


# Data generator with MFCC features and decoded audio for mel spectrogram
for i in range(nr_batches):
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder=os.path.join(DATA_PATH, FOLDER_NAME),
                                      save_audio=True,
                                      scale=False)

    generator.generate(train_size, file_prefix=f"train_")
    generator.generate(test_size, file_prefix=f"test_")

    # Decode audio files and save as numpy arrays
    train_audio = tf.stack([wav_to_binary(os.path.join(DATA_PATH, FOLDER_NAME, "audio", f"train_output_{j}.wav")) \
                            for j in range(train_size)], axis=0)
    train_audio_np = train_audio.numpy()
    np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_audio_decoded.npy"), train_audio_np)

    test_audio = tf.stack([wav_to_binary(os.path.join(DATA_PATH, FOLDER_NAME, "audio", f"test_output_{j}.wav")) \
                           for j in range(test_size)], axis=0)
    test_audio_np = test_audio.numpy()
    np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_audio_decoded.npy"), test_audio_np)

    # Load and resave generated feature files - UNSCALED
    trainMFCC = np.load(os.path.join(DATA_PATH, FOLDER_NAME, "train_features.npy"))
    np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_features.npy"), trainMFCC)
    testMFCC = np.load(os.path.join(DATA_PATH, FOLDER_NAME, "test_features.npy"))
    np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_features.npy"), testMFCC)

    # Load and resave generated patch files
    trainParams = np.load(os.path.join(DATA_PATH, FOLDER_NAME, "train_patches.npy"))
    np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_patches.npy"), trainParams)
    testParams = np.load(os.path.join(DATA_PATH, FOLDER_NAME, "test_patches.npy"))
    np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_patches.npy"), testParams)


# # Setup data generator for evaluation data audio files
# generator_eval = spgl.DatasetGenerator(synth, features,
#                                        output_folder=os.path.join(DATA_PATH, "evaluation_data"),
#                                        save_audio=True)
# generator_eval.generate(eval_size, file_prefix="eval_")


# Concatenate files and save
train_audio_decoded = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_audio_decoded.npy")) for i in range(nr_batches)], axis=0)
test_audio_decoded = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_audio_decoded.npy")) for i in range(nr_batches)], axis=0)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "train_audio_decoded.npy"), train_audio_decoded)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "test_audio_decoded.npy"), test_audio_decoded)

train_features = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_features.npy")) for i in range(nr_batches)], axis=0)
test_features = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_features.npy")) for i in range(nr_batches)], axis=0)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "train_features.npy"), train_features)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "test_features.npy"), test_features)

train_patches = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_patches.npy")) for i in range(nr_batches)], axis=0)
test_patches = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_patches.npy")) for i in range(nr_batches)], axis=0)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "train_patches.npy"), train_patches)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "test_patches.npy"), test_patches)