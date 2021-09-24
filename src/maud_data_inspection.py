#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../data/dataset_124params"

# Load data files
#trainAudio = np.load(os.path.join(DATA_PATH, "train_audio_decoded.npy"))
#trainMels = np.load(os.path.join(DATA_PATH, "train_melspectrogram.npy"))
#testAudio = np.load(os.path.join(DATA_PATH, "test_audio_decoded.npy"))
testMels = np.load(os.path.join(DATA_PATH, "test_melspectrogram.npy"))

# Check dimensions of training and test data
#print(f"The shape of trainAudio: {trainAudio.shape}")
#print(f"The shape of trainMels: {trainMels.shape}")
#print(f"The shape of testAudio: {testAudio.shape}")
print(f"The shape of testMels: {testMels.shape}")


# Plot decoded audio data
def plot_audio(array_file, index:int):
    audio = array_file[index].astype("float32")

    plt.plot(audio) # x = time; y = amplitude (loudness)
    plt.show()
    plt.close()


# Plot mel spectrogram
def plot_mel(array_file, index:int):
    melspectrogram = array_file[index]

    plt.imshow(melspectrogram[:,:,0], cmap="inferno")
    plt.axis("off")
    plt.show()
    plt.close()


