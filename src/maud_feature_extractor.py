#!/usr/bin/env python
import os
import librosa
import numpy as np
import tensorflow as tf


DATA_PATH = "../data/dataset_124params"
TRAIN_PATH = "../data/dataset_124params/train_dataset2"
VAL_PATH = "../data/dataset_124params/val_dataset2"

# If training and validation dataset folders do not exist
if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH)

if not os.path.exists(VAL_PATH):
    os.makedirs(VAL_PATH)

# Transform decoded audio into mel spectrogram - with tensorflow
# SEEMS SLOWER BECAUSE OF MULTIPLE OPERATIONS
# WITH TENSORFLOW_IO THIS COULD BE FASTER BUT CANNOT USE WITH MAC M1
def tf_audio_to_mel(audio):

    def tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def power_to_db(magnitude, amin=1e-16, top_db=80.0):
        ref_value = tf.reduce_max(magnitude)
        log_spec = 10.0 * tf_log10(tf.maximum(amin, magnitude))
        log_spec -= 10.0 * tf_log10(tf.maximum(amin, ref_value))
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
        return log_spec

    spectrogram = tf.signal.stft(audio,
                                  frame_length=4096,
                                  fft_length=4096,
                                  frame_step=256,
                                  pad_end=True)

    magnitude_spectrogram = tf.abs(spectrogram)

    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=256,
        num_spectrogram_bins=4096 // 2 + 1,
        sample_rate=44100,
        lower_edge_hertz=0,
        upper_edge_hertz=20000)

    mel_spectrogram = tf.matmul(tf.square(magnitude_spectrogram), mel_filterbank)

    # convert to decibel-scale melspectrogram: compute dB relative to peak power
    log_mel_spectrogram = power_to_db(mel_spectrogram)

    # normalize data
    log_mel_spectrogram = tf.divide(tf.subtract(log_mel_spectrogram, tf.math.reduce_min(log_mel_spectrogram)), \
                                     tf.subtract(tf.math.reduce_max(log_mel_spectrogram), tf.math.reduce_min(log_mel_spectrogram)))
    # add dimension for channel
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)
    # change order to frequency, time, channel
    log_mel_spectrogram = tf.transpose(log_mel_spectrogram, perm=(1, 0 ,2)) # Change order of axes
    # flip frequency axis so low frequencies are at bottom of image
    log_mel_spectrogram = log_mel_spectrogram[::-1, :, :]

    return log_mel_spectrogram


# Transform decoded audio into mel spectrogram - with librosa
def lib_audio_to_mel(audio):
    audio = audio.astype("float32")

    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        n_fft=2048,
        win_length=2048,
        n_mels=128,
        hop_length=512,
        sr=44100,
        fmin=0,
        fmax=20000,
        center=True,
        power=2.0
    )

    # convert to decibel-scale melspectrogram: compute dB relative to peak power
    spectrogram = librosa.core.power_to_db(spectrogram, ref=np.max)
    # normalize data
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    # add dimension for channel
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    # flip frequency axis so low frequencies are at bottom of image
    spectrogram = spectrogram[::-1, :, :]

    return spectrogram


# Load the decoded audio files
train_audio = np.load(os.path.join(DATA_PATH, "train_audio_decoded.npy"))
test_audio = np.load(os.path.join(DATA_PATH, "test_audio_decoded.npy"))

# OPTION 1: CREATING ONE BIG NUMPY ARRAY -----------------------
# Mel spectrograms train set
train_mel = np.stack([lib_audio_to_mel(train_audio[i]) for i in range(train_audio.shape[0])], axis=0)
print(train_mel.shape)
np.save(os.path.join(DATA_PATH, "train_melspectrogram2.npy"), train_mel)

# Mel spectrograms test set
test_mel = np.stack([lib_audio_to_mel(test_audio[i]) for i in range(test_audio.shape[0])], axis=0)
print(test_mel.shape)
np.save(os.path.join(DATA_PATH, "test_melspectrogram2.npy"), test_mel)


# OPTION 2: CREATING NUMPY ARRAY FOR EACH EXAMPLE -----------------------
# Load the target files
train_patches = np.load(os.path.join(DATA_PATH, "train_patches.npy"))
test_patches = np.load(os.path.join(DATA_PATH, "test_patches.npy"))

# Mel spectrograms train set
for i in range(train_audio.shape[0]):
    train_mel = lib_audio_to_mel(train_audio[i])
    np.save(os.path.join(TRAIN_PATH, f"train_melspectrogram_{i}.npy"), train_mel)
    train_patch = train_patches[i]
    np.save(os.path.join(TRAIN_PATH, f"train_patches_{i}.npy"), train_patch)

# Mel spectrograms test set
for i in range(test_audio.shape[0]):
    test_mel = lib_audio_to_mel(test_audio[i])
    np.save(os.path.join(VAL_PATH, f"test_melspectrogram_{i}.npy"), test_mel)
    test_patch = test_patches[i]
    np.save(os.path.join(VAL_PATH, f"test_patches_{i}.npy"), test_patch)





