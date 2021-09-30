#!/usr/bin/env python
import matplotlib.pyplot as plt


# Plot decoded audio data
def plot_audio(audio):
    """
    Args:
        audio: Numpy array with decoded audio

    Returns: Visualisation of audio

    """

    audio = audio.astype("float32")
    plt.plot(audio) # x = time; y = amplitude (loudness)


# Plot mel spectrogram
def plot_mel(spectrogram):
    """
    Args:
        spectrogram: Numpy array with mel spectrogram

    Returns: Visualisation of the mel spectrogram

    """

    plt.imshow(spectrogram[:,:,0], cmap="inferno")
    plt.axis("off")