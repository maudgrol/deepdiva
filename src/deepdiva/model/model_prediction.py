#!/usr/bin/env python
import click
import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile
from deepdiva.data.dataset_generator import DatasetGenerator
from deepdiva.model.lstm_model import LstmHighwayModel
from deepdiva.model.cnn_model import ConvModel
from deepdiva.utils.patch_utils import split_train_override_patch
from deepdiva.utils.h2p_utils import H2P


@click.command()
@click.option('--data-path', 'data_path', default="./data/dataset", required=False,
              type=click.Path(exists=True), show_default=True, help='Path to data folder')
@click.option('--audio-path', 'audio_path', default="./data/dataset/audio", required=False,
              type=click.Path(exists=True), show_default=True, help='Path to audio folder')
@click.option('--audio_file', 'audio_file', default="all", required=False, type=str,
              show_default=True, help='Specific input .wav file or all .wav files in audio folder')
@click.option('--vst-path', 'vst_path', default="/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
              required=False, type=click.Path(exists=True), show_default=True, help='Path to vst plugin')
@click.option('--model-file', 'model_file', required=False, type=click.Path(exists=True),
              show_default=False, help='Path to model file or model weights checkpoint')
@click.option('--load_type', 'load_type', type=click.Choice(['model', 'weights'], case_sensitive=False),
              default='model', required=True, show_default=True, help="Load model object or model weights")
@click.option('--base-preset', 'base_preset', default="MS-REV1_deepdiva.h2p", required=False,
              type=str, show_default=True, help='DIVA preset that serves as base for fixed parameters')
@click.option('--random-parameters', 'random_parameters', type=str, default="",
              show_default=True, help="Indices of to be randomized parameters. Format: 'id id'")
@click.option('--sample-rate', 'sample_rate', default=44100, required=False,
              type=int, show_default=True, help='Sample rate for audio')
@click.option('--midi-note-pitch', 'midi_note_pitch', default=48, required=False,
              type=int, show_default=True, help='Midi note (C3 is 48)')
@click.option('--midi-note-velocity', 'midi_note_velocity', default=127,
              required=False, type=int, show_default=True, help='Midi note velocity (0-127)')
@click.option('--note-length-seconds', 'note_length_seconds', default=2.0,
              required=False, type=float, show_default=True, help='Note length in seconds')
@click.option('--render-length-seconds', 'render_length_seconds', default=2.0,
              required=False, type=float, show_default=True, help='Rendered audio length in seconds')
@click.option('--feature', 'feature', type=click.Choice(['spectrogram', 'mfcc'], case_sensitive=False),
              required=True, show_default=True, help="Which type of feature to extract")
@click.option('--scaler-file', 'scaler_file', default=None,
              required=False, type=str, show_default=True, help='File name of saved data scaler object')
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
@click.option('--fmin', 'freq_min', default=50, required=False, type=int,
              show_default=True, help='Lowest frequency (in Hz)')
@click.option('--fmax', 'freq_max', default=15000, required=False, type=int,
              show_default=True, help='Highest frequency (in Hz)')
@click.option('--time-major/--no-time-major', 'time_major', default=True,
              show_default=True, help='Change MFCC to shape (time_slices, n_mfcc) for modelling')


def click_main(data_path, audio_path, audio_file, vst_path, model_file, load_type, base_preset, random_parameters,
               sample_rate, midi_note_pitch, midi_note_velocity, note_length_seconds, render_length_seconds, feature,
               scaler_file, n_fft, win_length, hop_length, n_mels, n_mfcc, freq_min, freq_max, time_major):
    """
    Interface for Click CLI.
    """

    main(data_path=data_path, audio_path=audio_path, audio_file=audio_file, vst_path=vst_path, model_file=model_file,
         load_type=load_type, base_preset=base_preset, random_parameters=random_parameters, sample_rate=sample_rate,
         midi_note_pitch=midi_note_pitch, midi_note_velocity=midi_note_velocity, note_length_seconds=note_length_seconds,
         render_length_seconds=render_length_seconds, feature=feature, scaler_file=scaler_file, n_fft=n_fft,
         win_length=win_length, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc, freq_min=freq_min,
         freq_max=freq_max, time_major=time_major)


def main(data_path, audio_path, audio_file, vst_path, model_file, load_type, base_preset, random_parameters,
               sample_rate, midi_note_pitch, midi_note_velocity, note_length_seconds, render_length_seconds, feature,
               scaler_file, n_fft, win_length, hop_length, n_mels, n_mfcc, freq_min, freq_max, time_major):
    """Runs model prediction script """

    if audio_file == "all":
        # Decode all audio files in audio folder
        audio = np.stack([wav_to_audio(os.path.join(audio_path, file), sample_rate) for file in os.listdir(audio_path) if file.endswith(".wav") and "prediction" not in file], axis=0)
    else:
        audio = wav_to_audio(os.path.join(audio_path, audio_file), sample_rate)

    # Extract features
    extractor = FeatureExtractor(data_path=data_path, saved_scaler=True, scaler_file=scaler_file)

    if feature == "spectrogram":
        features = np.stack([extractor.melspectrogram(audio=audio[i], n_fft=n_fft, win_length=win_length,
                                                         hop_length=hop_length, n_mels=n_mels, sample_rate=sample_rate,
                                                         freq_min=freq_min, freq_max=freq_max) \
                                for i in range(audio.shape[0])], axis=0)

    if feature == "mfcc":
        features = np.stack([extractor.mfcc(audio=audio[i], n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                        n_mfcc=n_mfcc, sample_rate=sample_rate, freq_min=freq_min, freq_max=freq_max,
                                        time_major=time_major) \
                         for i in range(audio.shape[0])], axis=0)

    np.save(os.path.join(data_path, "eval_features.npy"), features)


# Decode .wav file to audio
def wav_to_audio(file, sample_rate):
    # Loading and decoding the wav file.
    audio, _ = librosa.load(file, sr=sample_rate, mono=True, offset=0.0, duration=2.0)

    audio = audio.astype("float32")

    return audio


if __name__ == '__main__':

    click_main()