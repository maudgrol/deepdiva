#!/usr/bin/env python
import click
import os
import numpy as np
import matplotlib.pyplot as plt
from deepdiva.utils.visualisation_utils import *

@click.command()
@click.option('--data-path', 'data_path', default="./data/dataset", required=False,
              type=click.Path(exists=True), show_default=True, help='Path to data folder')
@click.option('--show-audio/--no-show-audio', 'show_audio', default=False,
              show_default=True, help='Whether to visualize the audio')
@click.option('--audio-file', 'audio_file', default="audio.npy",
              required=False, type=str, show_default=True, help='File name for audio dataset (.npy)')
@click.option('--show-spectrogram/--no-show-spectrogram', 'show_spectrogram', default=False,
              show_default=True, help='Whether to visualize the mel spectrogram')
@click.option('--spectrogram-file', 'spectrogram_file', default="melspectrogram.npy",
              required=False, type=str, show_default=True, help='File name for mel spectrogram dataset (.npy)')


def click_main(data_path, audio_file, spectrogram_file, show_spectrogram, show_audio):
    """
    Interface for Click CLI.
    """

    main(data_path=data_path, audio_file=audio_file, spectrogram_file=spectrogram_file,
         show_spectrogram=show_spectrogram, show_audio=show_audio)


def main(data_path, audio_file, spectrogram_file, show_spectrogram, show_audio):
    """Runs data visualisation scripts to inspect data"""

    if not show_audio and not show_spectrogram:
        print("Please select at least one type of plot: audio or mel spectrogram")

    if show_audio:
        # Load data file
        audio = np.load(os.path.join(data_path, audio_file))
        print(f"The shape of the audio file: {audio.shape}")

        # Plot 6 audio files
        fig1 = plt.figure(num="Audio signal", figsize=(12, 6))

        for sp in range(6):
            plt.subplot(2, 3, sp+1)
            sample = np.random.choice(audio.shape[0])
            plot_audio(audio[sample])
            plt.title(f"sample: {sample}")

        plt.show()
        plt.close()


    if show_spectrogram:
        # Load data file
        spectrogram = np.load(os.path.join(data_path, spectrogram_file))
        print(f"The shape of the mel spectrogram file: {spectrogram.shape}")

        # Plot 6 mel spectrograms
        fig2 = plt.figure(num="Mel spectrogram", figsize=(12, 6))

        for sp in range(6):
            plt.subplot(2, 3, sp+1)
            sample = np.random.choice(spectrogram.shape[0])
            plot_mel(spectrogram[sample])
            plt.title(f"sample: {sample}")

        plt.show()
        plt.close()


if __name__ == '__main__':

    click_main()
