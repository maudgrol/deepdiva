#!/usr/bin/env python
import os
import numpy as np
import pickle
import librosa


class FeatureExtractor():

    def __init__(self,
                 data_path,
                 saved_scaler,
                 scaler_file):

        """
        Constructor
        """

        self.data_path = data_path
        self.saved_scaler = saved_scaler
        self.scaler_file = scaler_file


    def melspectrogram(self, audio, n_fft, win_length, hop_length,
                       n_mels, sample_rate, freq_min, freq_max):
        """
        :param audio: audio time-series, numpy array
        :param n_fft: number of Fast Fourier Transform components
        :param win_length: each frame of audio is windowed by window() and
                           will be of length win_length and then padded with zeros
        :param hop_length: number of samples between successive frames
        :param n_mels: number of Mel bands to return
        :param sample_rate: sampling rate of the incoming signal
        :param freq_min: lowest frequency (in Hz)
        :param freq_max: highest frequency (in Hz)
        :return: Normalized, decibel-scaled mel spectrogram with shape (n_mels, time_slices, channel)
        """
        # Assert audio file is specified
        assert audio is not None, "Please specify the audio file"

        # Assert number of data mel frequency bands is specified
        assert n_mels is not None, "Please specify the number of Mel bands to return"

        # If win_length is unspecified, defaults to n_fft = win_length
        if win_length is None:
            win_length = n_fft

        # If highest frequency (in Hz) is unspecified, defaults to sample rate / 2
        if freq_max is None:
            freq_max = sample_rate // 2

        audio = audio.astype("float32")

        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            sr=sample_rate,
            fmin=freq_min,
            fmax=freq_max,
            center=True,
            power=2.0
        )

        # convert to decibel-scale melspectrogram: compute dB relative to peak power
        spectrogram = librosa.core.power_to_db(spectrogram, ref=np.max)
        # normalize data
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
        # add dimension for channel
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        # flip frequency axis so low frequencies are at bottom of image
        spectrogram = spectrogram[::-1, :, :]

        return spectrogram


    def mfcc(self, audio, n_fft, win_length, hop_length, n_mfcc,
             sample_rate, freq_min, freq_max, time_major):
        """
        :param audio: audio time-series, numpy array
        :param n_fft: number of Fast Fourier Transform components
        :param win_length: each frame of audio is windowed by window() and
                           will be of length win_length and then padded with zeros
        :param hop_length: number of samples between successive frames
        :param n_mfcc: number of MFCCs to return
        :param sample_rate: sampling rate of the incoming signal
        :param freq_min: lowest frequency (in Hz)
        :param freq_max: highest frequency (in Hz)
        :param time_major: change data to shape (time_slices, n_mfcc) for modelling
        :return: Normalized MFCCs with shape (time_slices, n_mfcc)
        """

        # Assert audio file is specified
        assert audio is not None, "Please specify the audio file"

        # Assert number of data mel frequency bands is specified
        assert n_mfcc is not None, "Please specify the number of MFCCs to return"

        # Assert that the saved scaler object exists if used
        if self.saved_scaler:
            assert self.scaler_file is not None, "Please specify scaler file when using previously saved scaler object"
            assert os.path.exists(os.path.join(self.data_path, self.scaler_file)), "The specified scaler file does not exist"

        # If win_length is unspecified, defaults to n_fft = win_length
        if win_length is None:
            win_length = n_fft

        # If highest frequency (in Hz) is unspecified, defaults to sample rate / 2
        if freq_max is None:
            freq_max = sample_rate // 2

        audio = audio.astype("float32")

        mfcc = librosa.feature.mfcc(
            y=audio,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mfcc=n_mfcc,
            sr=sample_rate,
            fmin=freq_min,
            fmax=freq_max
        )

        # Change data to shape (time_slices, n_mfcc) for modelling
        if time_major:
            mfcc = np.transpose(mfcc)

        # Normalize data based on saved data_scaler object
        if self.saved_scaler:
            mfcc_scaler = self.__load_data_scaler()

            mfcc = (mfcc - mfcc_scaler["mfcc_min"]) / mfcc_scaler["mfcc_range"]

        return mfcc


    def __load_data_scaler(self):
        # Load previously saved data_scaler.pickle file
        with open(os.path.join(self.data_path, self.scaler_file), 'rb') as handle:
            data_scaler = pickle.load(handle)

        return data_scaler







