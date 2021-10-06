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
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.tight_layout()


# Plot mel spectrogram
def plot_mel(spectrogram):
    """
    Args:
        spectrogram: Numpy array with mel spectrogram

    Returns: Visualisation of the mel spectrogram

    """

    plt.imshow(spectrogram[:,:,0], cmap="inferno")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("time")
    plt.ylabel("mel bands")


# Plot loss and validation loss after model training
def plot_loss(history):
    """
    Args:
        history: tf.keras.callbacks.History

    Returns: Visualisation of the loss and validation loss across training epochs

    """

    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    plt.show()
    plt.close()
