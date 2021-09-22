#!/usr/bin/env python
import os
import librosa
import numpy as np
import tensorflow as tf


DATA_PATH = "../data/dataset_4params"

# Transform decoded audio into mel spectrogram
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

    spectrograms = tf.signal.stft(audio,
                                  frame_length=4096,
                                  fft_length=4096,
                                  frame_step=256,
                                  pad_end=True)

    magnitude_spectrograms = tf.abs(spectrograms)

    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=256,
        num_spectrogram_bins=4096 // 2 + 1,
        sample_rate=44100,
        lower_edge_hertz=0,
        upper_edge_hertz=20000)

    mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms), mel_filterbank)

    log_mel_spectrograms = power_to_db(mel_spectrograms)

    log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, axis=-1) # 2D -> 3D. Image resize requires 3 dimensional input
    log_mel_spectrograms = tf.transpose(log_mel_spectrograms, perm=(1, 0 ,2)) # Change order of axes
    log_mel_spectrograms = log_mel_spectrograms[::-1, :, :] # Flip the first axis (frequency)

    return log_mel_spectrograms


# Using librosa to preprocess audio data - not used in project??
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

    spectrogram = librosa.core.power_to_db(spectrogram, ref=np.min)

    spectrogram /= np.max(spectrogram)  # normalization: spectrogram / np.max
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # add axis at end
    spectrogram = spectrogram[::-1, :, :] # we flip the image upside down so lower tones are lower in image

    return spectrogram


# Load the decoded audio files
train_audio = np.load(os.path.join(DATA_PATH, "train_audio_decoded.npy"))
# test_audio = np.load(os.path.join(DATA_PATH, "test_audio_decoded.npy"))

# Mel spectrograms train set - not necessary to process data offline?
train_mel = tf.stack([tf_audio_to_mel(train_audio[i]) for i in range(train_audio.shape[0])], axis=0)
print(train_mel.shape)
train_mel.numpy()
np.save(os.path.join(DATA_PATH, "train_melspectrogram.npy"), train_mel)


# # Mel spectrograms test set
# test_mel = tf.stack([tf_audio_to_mel(test_audio[i]) for i in range(test_audio.shape[0])], axis=0)
# print(test_mel.shape)
# test_mel.numpy()
# np.save(os.path.join(DATA_PATH, "test_melspectrogram.npy"), test_mel)





