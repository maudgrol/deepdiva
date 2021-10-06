#!/usr/bin/env python
import click
import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile
import copy
import librenderman as rm
import scipy.io.wavfile
from deepdiva.features.feature_extractor import FeatureExtractor
from deepdiva.model.lstm_model import LstmHighwayModel
from deepdiva.model.cnn_model import ConvModel
from deepdiva.utils.patch_utils import split_train_override_patch
from deepdiva.utils.h2p_utils import H2P
from deepdiva.utils.model_utils import root_mean_squared_error


@click.command()
@click.option('--data-path', 'data_path', default="./data", required=False,
              type=click.Path(exists=True), show_default=True, help='Path to data folder')
@click.option('--audio-path', 'audio_path', default="./data/dataset/audio", required=False,
              type=click.Path(exists=True), show_default=True, help='Path to audio folder')
@click.option('--audio-file', 'audio_file', required=True, type=str,
              show_default=True, help='Input .wav file in audio folder')
@click.option('--vst-path', 'vst_path', default="/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
              required=False, type=click.Path(exists=True), show_default=True, help='Path to vst plugin')
@click.option('--model-file', 'model_file', required=False, type=click.Path(exists=True),
              default="./models/cnn_melspectrogram_11params", show_default=True, help='Path to model file')
@click.option('--base-preset', 'base_preset', default="MS-REV1_deepdiva.h2p", required=False,
              type=str, show_default=True, help='DIVA preset that serves as base for fixed parameters')
@click.option('--random-parameters', 'random_parameters', type=str, default= "33 34 35 86 87 97 98 131 132 148 149",
              show_default=True, help="Indices of to be randomized parameters (ascending order). Format: 'id id'")
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
              default='spectrogram', required=False, show_default=True, help="Which type of feature to extract")
@click.option('--scaler-file', 'scaler_file', default="train_mfcc_scaling.pickle",
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


def click_main(data_path, audio_path, audio_file, vst_path, model_file, base_preset, random_parameters,
               sample_rate, midi_note_pitch, midi_note_velocity, note_length_seconds, render_length_seconds, feature,
               scaler_file, n_fft, win_length, hop_length, n_mels, n_mfcc, freq_min, freq_max, time_major):
    """
    Interface for Click CLI.
    """
    # Create a list of integers of parameters to randomize from random_parameters string variable
    random_parameters = random_parameters.split()
    random_parameters = [int(x) for x in random_parameters]

    main(data_path=data_path, audio_path=audio_path, audio_file=audio_file, vst_path=vst_path, model_file=model_file,
         base_preset=base_preset, random_parameters=random_parameters, sample_rate=sample_rate,
         midi_note_pitch=midi_note_pitch, midi_note_velocity=midi_note_velocity, note_length_seconds=note_length_seconds,
         render_length_seconds=render_length_seconds, feature=feature, scaler_file=scaler_file, n_fft=n_fft,
         win_length=win_length, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc, freq_min=freq_min,
         freq_max=freq_max, time_major=time_major)


def main(data_path, audio_path, audio_file, vst_path, model_file, base_preset, random_parameters,
               sample_rate, midi_note_pitch, midi_note_velocity, note_length_seconds, render_length_seconds, feature,
               scaler_file, n_fft, win_length, hop_length, n_mels, n_mfcc, freq_min, freq_max, time_major):
    """Runs model prediction script """

    audio = wav_to_audio(os.path.join(audio_path, audio_file), sample_rate, render_length_seconds)

    # Extract features
    extractor = FeatureExtractor(data_path=data_path, saved_scaler=True, scaler_file=scaler_file)

    if feature == "spectrogram":
        features = extractor.melspectrogram(audio=audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                            n_mels=n_mels, sample_rate=sample_rate, freq_min=freq_min,
                                            freq_max=freq_max)

    if feature == "mfcc":
        features = extractor.mfcc(audio=audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                  n_mfcc=n_mfcc, sample_rate=sample_rate, freq_min=freq_min, freq_max=freq_max,
                                  time_major=time_major)

    # Load final model or model weigths
    # model = tf.keras.models.load_model(model_file)
    model = ConvModel(shape=(128, 173, 1), num_outputs=11)
    model.load_weights(os.path.join(model_file, "cp-0075.ckpt")).expect_partial()

    # Model prediction - add a dimension for samples
    prediction = model.predict(tf.expand_dims(features, axis=0))[0].astype("float64")

    # Use the predicted values for randomized parameters
    predicted_parameters = list(zip(random_parameters, prediction))

    # Join predicted parameters with overridden parameters that are based on base preset
    used_base_preset = os.path.join(data_path, base_preset)
    h2p = H2P()
    base_patch = h2p.preset_to_patch(h2p_filename=used_base_preset)

    overridden_parameters, _ = split_train_override_patch(base_patch, random_parameters)

    overridden_parameters.extend(predicted_parameters)
    overridden_parameters.sort()
    full_predicted_patch = copy.deepcopy(overridden_parameters)

    # Save preset from the predicted patch
    preset = h2p.patch_to_preset(patch=full_predicted_patch,
                                 h2p_filename=os.path.join(data_path, f"predicted_preset.h2p"))


    # Render and save audio of the predicted patch
    patch_to_wav(audio_path=audio_path, audio_file=audio_file, vst_path=vst_path, patch=full_predicted_patch,
                 sample_rate=sample_rate, midi_note_pitch=midi_note_pitch, midi_note_velocity=midi_note_velocity,
                 note_length_seconds=note_length_seconds, render_length_seconds=render_length_seconds)

    print(f"The created DIVA preset 'predicted_preset.h2p' was saved in {data_path}")
    print(f"The predicted sound file 'predicted_{audio_file}' was saved in {audio_path}")


def wav_to_audio(file, sample_rate, render_length_seconds):
    """
    Loading and decoding the input wav file
    """
    audio, _ = librosa.load(file, sr=sample_rate, mono=True, offset=0.0, duration=render_length_seconds)

    audio = audio.astype("float32")

    return audio


def patch_to_wav(audio_path, audio_file, vst_path, patch, sample_rate, midi_note_pitch, midi_note_velocity, note_length_seconds,
                 render_length_seconds):
    """
    Render audio from predicted patch and save .wav file
    """
    # Configure synthesizer
    engine = rm.RenderEngine(sample_rate, 512, 512)
    engine.load_plugin(vst_path)

    # Set patch to predicted patch
    engine.set_patch(patch)

    # Render the patch
    engine.render_patch(midi_note_pitch, midi_note_velocity, note_length_seconds, render_length_seconds)

    # Get decoded audio
    audio = engine.get_audio_frames()
    audio = np.array(audio, copy=True, dtype=np.float32)

    # Save .wav file based on audio
    scipy.io.wavfile.write(os.path.join(audio_path, f"predicted_{audio_file}"), sample_rate, audio)


if __name__ == '__main__':

    click_main()