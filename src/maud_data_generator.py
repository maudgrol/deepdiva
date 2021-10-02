#!/usr/bin/env python
from utils.patch_utils import preset_to_patch, split_train_override_patch
import spiegelib as spgl
import os
import numpy as np
import tensorflow as tf
import soundfile

DATA_PATH = "../data"
VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
FOLDER_NAME = "dataset_124params"
BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"

# Generated samples per data batch
TRAIN_SIZE = 1000
TEST_SIZE = 200
EVAL_SIZE = 20
# Number of data batches
NR_BATCHES = 50
COMPLETED = 0

PARAMETERS_TO_RANDOMIZE = []
PARAMETERS_TO_RANDOMIZE.extend(range(4, 16))
PARAMETERS_TO_RANDOMIZE.extend([33, 34, 35])
PARAMETERS_TO_RANDOMIZE.extend(range(37, 43))
PARAMETERS_TO_RANDOMIZE.extend([44, 45, 46])
PARAMETERS_TO_RANDOMIZE.extend(range(48, 54))
PARAMETERS_TO_RANDOMIZE.extend(range(85, 143))
PARAMETERS_TO_RANDOMIZE.extend(range(144, 154))
PARAMETERS_TO_RANDOMIZE.extend(range(155, 167))
PARAMETERS_TO_RANDOMIZE.extend([169, 170, 171, 174])
PARAMETERS_TO_RANDOMIZE.extend(range(264, 271))
PARAMETERS_TO_RANDOMIZE.extend([278, 279, 280])

# use base patch for overwritten parameter settings
base_patch = preset_to_patch(BASE_PATCH)
override_parameters, train_parameters = split_train_override_patch(base_patch, PARAMETERS_TO_RANDOMIZE)


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
for i in range(COMPLETED, NR_BATCHES):
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder=os.path.join(DATA_PATH, FOLDER_NAME),
                                      save_audio=True,
                                      scale=True)

    generator.generate(TRAIN_SIZE, file_prefix=f"train_")
    generator.generate(TEST_SIZE, file_prefix=f"test_")
    generator.save_scaler(f"data_scaler_{i}.pkl")

    #Decode audio files and save as numpy arrays
    train_audio = tf.stack([wav_to_binary(os.path.join(DATA_PATH, FOLDER_NAME, "audio", f"train_output_{j}.wav")) \
                            for j in range(TRAIN_SIZE)], axis=0)
    train_audio_np = train_audio.numpy()
    np.save(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_audio_decoded.npy"), train_audio_np)

    test_audio = tf.stack([wav_to_binary(os.path.join(DATA_PATH, FOLDER_NAME, "audio", f"test_output_{j}.wav")) \
                           for j in range(TEST_SIZE)], axis=0)
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


# Setup data generator for evaluation data audio files
generator_eval = spgl.DatasetGenerator(synth, features,
                                       output_folder=os.path.join(DATA_PATH, "eval_dataset_124params"),
                                       save_audio=True)
generator_eval.generate(EVAL_SIZE, file_prefix="eval_")


# Concatenate files and save
train_audio_decoded = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_audio_decoded.npy")) for i in range(NR_BATCHES)], axis=0)
test_audio_decoded = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_audio_decoded.npy")) for i in range(NR_BATCHES)], axis=0)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "train_audio_decoded.npy"), train_audio_decoded)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "test_audio_decoded.npy"), test_audio_decoded)

train_features = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_features.npy")) for i in range(NR_BATCHES)], axis=0)
test_features = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_features.npy")) for i in range(NR_BATCHES)], axis=0)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "train_features.npy"), train_features)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "test_features.npy"), test_features)

train_patches = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"train_{i}_patches.npy")) for i in range(NR_BATCHES)], axis=0)
test_patches = np.concatenate([np.load(os.path.join(DATA_PATH, FOLDER_NAME, f"test_{i}_patches.npy")) for i in range(NR_BATCHES)], axis=0)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "train_patches.npy"), train_patches)
np.save(os.path.join(DATA_PATH, FOLDER_NAME, "test_patches.npy"), test_patches)