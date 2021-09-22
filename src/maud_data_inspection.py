#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../data/dataset_4params"

# Load data files
#trainAudio = np.load(os.path.join(DATA_PATH, "train_audio_decoded.npy"))
trainMels = np.load(os.path.join(DATA_PATH, "train_melspectrogram.npy"))
#trainMFCC = np.load(os.path.join(DATA_PATH, "train_features.npy"))

#testAudio = np.load(os.path.join(DATA_PATH, "test_audio_decoded.npy"))
testMels = np.load(os.path.join(DATA_PATH, "test_melspectrogram.npy"))
#testMFCC = np.load(os.path.join(DATA_PATH, "test_features.npy"))

# Check dimensions of training and test data
#print(f"The shape of trainAudio: {trainAudio.shape}")
print(f"The shape of trainMels: {trainMels.shape}")
#print(f"The shape of trainMFCC: {trainMFCC.shape}")

#print(f"The shape of testAudio: {testAudio.shape}")
print(f"The shape of testMels: {testMels.shape}")
#print(f"The shape of testMFCC: {testMFCC.shape}")


# Plot decoded audio data
def plot_audio(array_file, index:int):
    audio = array_file[index].astype("float32")

    plt.plot(audio) # x = time; y = amplitude (loudness)
    plt.show()
    plt.close()


# Plot mel spectrogram
def plot_mel(array_file, index: int):
    mel = array_file[index]
    mel = mel[:,:,0].astype("float32")

    plt.imshow(mel, cmap="inferno")
    plt.axis("off")
    plt.show()
    plt.close()


# Plot Mel-Frequency Cepstral Coefficients
def plot_mfcc(filepath, array_file: str, index: int):
    mfcc = np.load(os.path.join(filepath, array_file))
    mfcc = mfcc[index].astype("float32")

    #plt.imshow(mfcc, cmap="inferno")
    plt.plot(mfcc)
    plt.show()
    plt.close()
