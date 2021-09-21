#!/usr/bin/env python
import os
import soundfile
import librosa
import cv2
import numpy as np
import tensorflow as tf


DATA_PATH = "../data/dataset"


# Using librosa to preprocess audio data
def audio_to_mel(audio):
    audio = audio[:,0].astype("float32")

    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        n_fft=1024,
        win_length=1024,
        n_mels=64,
        hop_length=64,
        sr=44100,
        fmax=8000
    )

    spectrogram /= np.max(spectrogram)  # normalization: spectrogram / np.max(audio)
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # add axis at end

    return spectrogram


# Load the decoded audio files
train_audio = np.load(os.path.join(DATA_PATH, "train_audio_decoded.npy"))
test_audio = np.load(os.path.join(DATA_PATH, "test_audio_decoded.npy"))

# Mel spectrograms train set
train_mel = np.stack([audio_to_mel(train_audio[i]) for i in range(train_audio.shape[0])], axis=0)
print(train_mel.shape)
np.save(os.path.join(DATA_PATH, "train_melspectrogram.npy"), train_mel)


# Mel spectrograms test set
test_mel = np.stack([audio_to_mel(test_audio[i]) for i in range(test_audio.shape[0])], axis=0)
print(test_mel.shape)
np.save(os.path.join(DATA_PATH, "test_melspectrogram.npy"), train_mel)





