#!/usr/bin/env python
import click
import os
import pickle
import numpy as np
from dotenv import find_dotenv, load_dotenv
from deepdiva.features.feature_extractor import FeatureExtractor


@click.command()
@click.option('--feature', 'feature', type=click.Choice(['spectrogram', 'mfcc'], case_sensitive=False),
              required=True, show_default=True, help="Which type of feature to extract")
@click.option('--data-path', 'data_path', default="./data/dataset", required=False,
              type=click.Path(exists=True), show_default=True, help='Path to dataset folder')
@click.option('--data-file', 'data_file', default=None, required=True, type=str,
              show_default=True, help='Audio file (file.npy) from which to extract features')
@click.option('--file-prefix', 'file_prefix', default="", required=False, type=str,
              show_default=True, help="Prefix for saving generated feature files (e.g. 'train_')")
@click.option('--saved-scaler/--no-saved-scaler', 'saved_scaler', default=False,
              show_default=True, help='Whether to use a previously saved data scaler object when extracting MFCCs')
@click.option('--scaler-file', 'scaler_file', default=None,
              required=False, type=str, show_default=True, help='File name of saved data scaler object')
@click.option('--scale-axis', 'scale_axis', default=0, required=False, show_default=True,
              help='Axis or axes to use for calculating scaling parameteres. Defaults to 0, which scales each MFCC and time series component independently.')
@click.option('--n_fft', 'n_fft', default=2048, required=False, type=int,
              show_default=True, help='Length of the FFT window')
@click.option('--win-length', 'win_length', default=None, required=False, type=int,
              show_default=True, help='Each frame of audio is windowed by window() and will be of length win_length and then padded with zeros. Defaults to win_length = n_fft')
@click.option('--hop_length', 'hop_length', default=512, required=False, type=int,
              show_default=True, help='Number of samples between successive frames')
@click.option('--n_mels', 'n_mels', default=128, required=False, type=int,
              show_default=True, help='Number of Mel bands to generate')
@click.option('--n_mfcc', 'n_mfcc', default=13, required=False, type=int,
              show_default=True, help='Number of MFCCs to return')
@click.option('--sample-rate', 'sample_rate', default=44100, required=False, type=int,
              show_default=True, help='Sampling rate of the incoming signal')
@click.option('--fmin', 'freq_min', default=50, required=False, type=int,
              show_default=True, help='Lowest frequency (in Hz)')
@click.option('--fmax', 'freq_max', default=15000, required=False, type=int,
              show_default=True, help='Highest frequency (in Hz)')
@click.option('--time-major/--no-time-major', 'time_major', default=True,
              show_default=True, help='Change MFCC to shape (time_slices, n_mfcc) for modelling')


def click_main(feature, data_path, data_file, file_prefix, saved_scaler, scaler_file, scale_axis,
               n_fft, win_length, hop_length, n_mels, n_mfcc, sample_rate, freq_min, freq_max, time_major):
    """
    Interface for Click CLI.
    """
    main(feature=feature, data_path=data_path, data_file=data_file, file_prefix=file_prefix,
         saved_scaler=saved_scaler, scaler_file=scaler_file, scale_axis=scale_axis, n_fft=n_fft,
         win_length=win_length, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc, sample_rate=sample_rate,
         freq_min=freq_min, freq_max=freq_max, time_major=time_major)


def main(feature, data_path, data_file, file_prefix, saved_scaler, scaler_file, scale_axis, n_fft,
         win_length, hop_length, n_mels, n_mfcc, sample_rate, freq_min, freq_max, time_major):
    """Runs data processing scripts to make feature set"""

    # Load audio data file
    audio = np.load(os.path.join(data_path, data_file))

    # Extract features
    extractor = FeatureExtractor(data_path=data_path, saved_scaler=saved_scaler, scaler_file=scaler_file)

    if feature == "spectrogram":
        spectrogram = np.stack([extractor.melspectrogram(audio=audio[i], n_fft=n_fft, win_length=win_length,
                                                        hop_length=hop_length, n_mels=n_mels, sample_rate=sample_rate,
                                                        freq_min=freq_min, freq_max=freq_max) \
                                 for i in range(audio.shape[0])], axis=0)

        # Save mel spectrogram feature set
        np.save(os.path.join(data_path, f"{file_prefix}melspectrogram.npy"), spectrogram)

    if feature == "mfcc":
        mfcc = np.stack([extractor.mfcc(audio=audio[i], n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                      n_mfcc=n_mfcc, sample_rate=sample_rate, freq_min=freq_min, freq_max=freq_max,
                                      time_major=time_major) \
                         for i in range(audio.shape[0])], axis=0)

        if not saved_scaler:
            # Normalize features
            mfcc_min = np.min(mfcc, axis=scale_axis)
            mfcc_range = np.max(mfcc, axis=scale_axis) - np.min(mfcc, axis=scale_axis)
            mfcc = (mfcc - mfcc_min) / mfcc_range

            # Save scaling dictionary
            mfcc_scaling = {"mfcc_min": mfcc_min, "mfcc_range": mfcc_range}
            with open(os.path.join(data_path, f"{file_prefix}mfcc_scaling.pickle"), 'wb') as handle:
                pickle.dump(mfcc_scaling, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save MFCC feature set
        np.save(os.path.join(data_path, f"{file_prefix}mfcc.npy"), mfcc)


if __name__ == '__main__':

    # Find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    click_main()
