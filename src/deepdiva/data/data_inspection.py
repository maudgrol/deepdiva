#!/usr/bin/env python
import click
import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import find_dotenv, load_dotenv
from deepdiva.utils.visualisation_utils import *

@click.command()
@click.option('--data-path', 'data_path', default="./data/dataset", required=False,
              type=click.Path(exists=True), show_default=True, help='Path to data folder')
@click.option('--show-audio/--no-show-audio', 'show_audio', default=False,
              show_default=True, help='Whether to visualize audio')
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

    if show_audio:
        # Load data file
        audio = np.load(os.path.join(data_path, audio_file))
        print(f"The shape of the audio file: {audio.shape}")

        # Plot 6 audio files
        fig1 = plt.figure(num=1, figsize=(12, 6))
        # Plot 1
        plt.subplot(2, 3, 1)
        plot_audio(audio[np.random.choice(audio.shape[0])])

        # Plot 2
        plt.subplot(2, 3, 2)
        plot_audio(audio[np.random.choice(audio.shape[0])])

        # Plot 3
        plt.subplot(2, 3, 3)
        plot_audio(audio[np.random.choice(audio.shape[0])])

        # Plot 4
        plt.subplot(2, 3, 4)
        plot_audio(audio[np.random.choice(audio.shape[0])])

        # Plot 5
        plt.subplot(2, 3, 5)
        plot_audio(audio[np.random.choice(audio.shape[0])])

        # Plot 6
        plt.subplot(2, 3, 6)
        plot_audio(audio[np.random.choice(audio.shape[0])])

        plt.show()
        plt.close()

    if show_spectrogram:
        # Load data file
        spectrogram = np.load(os.path.join(data_path, spectrogram_file))
        print(f"The shape of the mel spectrogram file: {spectrogram.shape}")

        # Plot 6 mel spectrograms
        fig2 = plt.figure(num=2, figsize=(12, 6))
        # Plot 1
        plt.subplot(2, 3, 1)
        plot_mel(spectrogram[np.random.choice(spectrogram.shape[0])])

        # Plot 2
        plt.subplot(2, 3, 2)
        plot_mel(spectrogram[np.random.choice(spectrogram.shape[0])])

        # Plot 3
        plt.subplot(2, 3, 3)
        plot_mel(spectrogram[np.random.choice(spectrogram.shape[0])])

        # Plot 4
        plt.subplot(2, 3, 4)
        plot_mel(spectrogram[np.random.choice(spectrogram.shape[0])])

        # Plot 5
        plt.subplot(2, 3, 5)
        plot_mel(spectrogram[np.random.choice(spectrogram.shape[0])])

        # Plot 6
        plt.subplot(2, 3, 6)
        plot_mel(spectrogram[np.random.choice(spectrogram.shape[0])])

        plt.show()
        plt.close()


if __name__ == '__main__':

    click_main()