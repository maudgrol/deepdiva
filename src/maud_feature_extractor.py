#!/usr/bin/env python
import os
import librosa
import numpy as np
import tensorflow as tf


DATA_PATH = "../data/dataset_4params"

# Transform decoded audio into mel spectrogram - with tensorflow SEEMS SLOWER
def tf_audio_to_mel(audio):
    audio = audio[:, 0]

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
    audio = audio[:,0].astype("float32")

    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        n_fft=4096,
        win_length=4096,
        n_mels=256,
        hop_length=256,
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
#train_audio = np.load(os.path.join(DATA_PATH, "train_audio_decoded.npy"))
test_audio = np.load(os.path.join(DATA_PATH, "test_audio_decoded.npy"))

# # Mel spectrograms train set
# # train_mel = np.stack([lib_audio_to_mel(train_audio[i]) for i in range(train_audio.shape[0])], axis=0)
# # print(train_mel.shape)
# # np.save(os.path.join(DATA_PATH, "train_melspectrogram.npy"), train_mel)
#
#
# Mel spectrograms test set
test_mel = np.stack([lib_audio_to_mel(test_audio[i]) for i in range(test_audio.shape[0])], axis=0)
print(test_mel.shape)
np.save(os.path.join(DATA_PATH, "test_melspectrogram.npy"), test_mel)





