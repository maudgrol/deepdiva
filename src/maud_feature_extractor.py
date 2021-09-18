#!/usr/bin/env python
import os
import soundfile
import librosa
import cv2
import numpy as np
import tensorflow as tf
#import tensorflow_io as tfio


DATA_PATH = "../data/toy_data"
AUDIO_PATH = "../data/toy_data/audio"

# Settings for mel spectrogram
IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH = 64, 64


def wav_to_binary(file):
    # Loading and decoding the wav file.
    audio_binary = tf.io.read_file(file)
    try:
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)
    except:
      data, samplerate = soundfile.read(file)
      soundfile.write(f"{file}_16.wav", data, samplerate, subtype='PCM_16')
      audio_binary = tf.io.read_file(f"{file}_16.wav")
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)

    audio = audio[:, 0]

    return audio


# Using librosa to preprocess audio data
def wav_to_mel(file):
    # Loading and decoding the wav file.
    audio_binary = tf.io.read_file(file)

    try:
        audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)
    except:
        data, samplerate = soundfile.read(file)
        soundfile.write(f"{file}_16.wav", data, samplerate, subtype='PCM_16')
        audio_binary = tf.io.read_file(f"{file}_16.wav")
        audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)

    audio = audio[:, 0]

    def py_preprocess_audio(audio):
        audio = audio.numpy().astype("float32")

        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            n_fft=1024,
            win_length=1024,
            n_mels=64,
            hop_length=64,
            sr=rate,
            fmax=10000
        )

        spectrogram /= np.max(spectrogram)  # normalization: spectrogram / np.max(audio)
        spectrogram = cv2.resize(spectrogram, dsize=(IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # add axis at end

        return spectrogram

    spectrogram = tf.py_function(py_preprocess_audio, [audio], tf.float32)
    spectrogram.set_shape((IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, 1))  # tell explicitly what shape of tensor is

    return spectrogram

# Using tensorflow-io to preprocess audio data (does not work with new Mac M1)
# def tf_wav_to_mel(file):
#   # Loading and decoding the wav file.
#   audio_binary = tf.io.read_file(file)
#   try:
#     audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)
#   except:
#     data, samplerate = soundfile.read(file)
#     soundfile.write(f"{file}_16.wav", data, samplerate, subtype='PCM_16')
#     audio_binary = tf.io.read_file(f"{file}_16.wav")
#     audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)
#
#   audio = audio[:, 0]
#
#   # Create spectrogram
#   spectrogram = tfio.audio.spectrogram(
#       input=audio,
#       nfft=1024, # nfft / 2 = number of frequency bins
#       window=1024, # nfft, window, stride determine resolution of mel spectrogram
#       stride=64,
#   )
#
#   # Create mel spectrogram
#   spectrogram = tfio.audio.melscale(
#       input=spectrogram,
#       rate=rate,
#       mels=64, # Number of mel frequency bins
#       fmin=0,
#       fmax=10000
#   )
#
#   spectrogram /= tf.math.reduce_max(spectrogram) # Normalize. Reduce_max finds absolute maximum in multidimensional array
#   spectrogram = tf.expand_dims(spectrogram, axis=-1) # 2D -> 3D. Image resize requires 3 dimensional input
#   spectrogram = tf.image.resize(spectrogram, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH)) # Resize image
#   spectrogram = tf.transpose(spectrogram, perm=(1, 0 ,2)) # Change order of axes
#   spectrogram = spectrogram[::-1, :, :] # Flip the first axis (frequency)
#
#   return spectrogram


# Create and save decoded audio and mel spectrogram data for training and validation - based on wav files
# Decoded audio train set - not so useful
train_audio = tf.stack([wav_to_binary(os.path.join(AUDIO_PATH, f"train_0_output_{i}.wav")) \
                        for i in range(10000)], axis=0)
print(train_audio.shape)

train_audio_np = train_audio.numpy()
np.save(os.path.join(DATA_PATH, "train_0_audio_decoded.npy"), train_audio_np)

# Decoded audio test set
test_audio = tf.stack([wav_to_binary(os.path.join(AUDIO_PATH, f"test_0_output_{i}.wav")) \
                       for i in range(1000)], axis=0)
print(test_audio.shape)

test_audio_np = test_audio.numpy()
np.save(os.path.join(DATA_PATH, "test_0_audio_decoded.npy"), test_audio_np)

# Mel spectrograms train set
train_mel = tf.stack([wav_to_mel(os.path.join(AUDIO_PATH, f"train_0_output_{i}.wav")) \
                      for i in range(10000)], axis=0)
print(train_mel.shape)

train_mel_np = train_mel.numpy()
np.save(os.path.join(DATA_PATH, "train_0_melspectrogram.npy"), train_mel_np)

# Mel spectrograms test set
test_mel = tf.stack([wav_to_mel(os.path.join(AUDIO_PATH, f"test_0_output_{i}.wav")) \
                     for i in range(1000)], axis=0)
print(test_mel.shape)

test_mel_np = test_mel.numpy()
np.save(os.path.join(DATA_PATH, "test_0_melspectrogram.npy"), test_mel_np)



