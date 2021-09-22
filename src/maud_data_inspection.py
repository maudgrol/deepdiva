#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_PATH = "../data/dataset_4params"

# Load data files
trainAudio = np.load(os.path.join(DATA_PATH, "train_audio_decoded.npy"))
#trainMels = np.load(os.path.join(DATA_PATH, "train_melspectrogram.npy"))

testAudio = np.load(os.path.join(DATA_PATH, "test_audio_decoded.npy"))
#testMels = np.load(os.path.join(DATA_PATH, "test_melspectrogram.npy"))

# Check dimensions of training and test data
print(f"The shape of trainAudio: {trainAudio.shape}")
#print(f"The shape of trainMels: {trainMels.shape}")

print(f"The shape of testAudio: {testAudio.shape}")
#print(f"The shape of testMels: {testMels.shape}")


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


# Plot decoded audio data
def plot_audio(array_file, index:int):
    audio = array_file[index].astype("float32")

    plt.plot(audio) # x = time; y = amplitude (loudness)
    plt.show()
    plt.close()


# Plot mel spectrogram
def plot_mel(array_file, index:int):
    audio = array_file[index]
    melspectrogram = tf_audio_to_mel(audio)

    plt.imshow(melspectrogram[:,:,0], cmap="inferno")
    plt.axis("off")
    plt.show()
    plt.close()


