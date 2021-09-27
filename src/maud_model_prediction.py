#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile
import spiegelib as spgl
from utils.patch_utils import preset_to_patch, patch_to_preset, split_train_override_patch
from maud_conv_model import ConvModel

DATA_PATH = "../data/eval_dataset_124params"
AUDIO_PATH = "../data/eval_dataset_124params/audio"
MODEL_PATH = "../models"
VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"

PARAMETERS_TO_RANDOMIZE = []
PARAMETERS_TO_RANDOMIZE.extend(range(4, 16))
PARAMETERS_TO_RANDOMIZE.extend([33, 34, 35])
PARAMETERS_TO_RANDOMIZE.extend(range(37, 43))
PARAMETERS_TO_RANDOMIZE.extend([44, 45, 46])
PARAMETERS_TO_RANDOMIZE.extend(range(48, 54))
PARAMETERS_TO_RANDOMIZE.extend(range(85, 143))
PARAMETERS_TO_RANDOMIZE.extend(range(144, 154))
PARAMETERS_TO_RANDOMIZE.extend(range(155, 167))
PARAMETERS_TO_RANDOMIZE.extend([169, 170, 171, 174])
PARAMETERS_TO_RANDOMIZE.extend(range(264, 271))
PARAMETERS_TO_RANDOMIZE.extend([278, 279, 280])

eval_audio = 0

with open(os.path.join(DATA_PATH, "mfcc_scaling.pickle"), 'rb') as handle:
    mfcc_scaler = pickle.load(handle)


# Import sound for evaluation and convert to mel spectrogram
# !!! Write function that also detects silence and trims?
def lib_wav_to_mel(file):
    # Loading and decoding the wav file.
    audio_binary = tf.io.read_file(file)
    try:
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)
    except:
      data, samplerate = soundfile.read(file)
      soundfile.write(file, data, samplerate, subtype='PCM_16')
      audio_binary = tf.io.read_file(file)
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)

    audio = audio[:, 0].numpy().astype("float32")

    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        n_fft=2048,
        win_length=2048,
        n_mels=128,
        hop_length=512,
        sr=rate,
        fmin=0,
        fmax=20000,
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


# !!! Write function that also detects silence and trims?
def lib_wav_to_mfcc(file, time_major=True):
    # Loading and decoding the wav file.
    audio_binary = tf.io.read_file(file)
    try:
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)
    except:
      data, samplerate = soundfile.read(file)
      soundfile.write(file, data, samplerate, subtype='PCM_16')
      audio_binary = tf.io.read_file(file)
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=-1)

    audio = audio[:, 0].numpy().astype("float32")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=44100,
        n_fft=2048,
        hop_length=1024,
        n_mfcc=13,
        fmin=0,
        fmax=20000
    )

    # Change data to format :(time_slices, features) for LSTM
    if time_major:
        mfcc = np.transpose(mfcc)

    # Normalize features
    mfcc = (mfcc - mfcc_scaler["mfcc_min"]) / mfcc_scaler["mfcc_range"]

    return mfcc


eval_mel = lib_wav_to_mel(os.path.join(AUDIO_PATH, f"eval_output_{eval_audio}.wav"))
print(eval_mel.shape)

eval_mfcc = lib_wav_to_mfcc(os.path.join(AUDIO_PATH, f"eval_output_{eval_audio}.wav"))
print(eval_mfcc.shape)

# load model
#model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "maud_training_26sep_08", "final_model_124params"))

model = ConvModel(shape=(256, 347, 1),
                  output_size=124)

latest = tf.train.latest_checkpoint(os.path.join(MODEL_PATH, "maud_training_26sep_08"))
model.load_weights(latest)

# add a dimension (for training samples) for prediction
prediction = model.predict(tf.expand_dims(eval_mel, axis=0))[0].astype("float64")

# use the predicted values for trainable parameters
pred = list(zip(PARAMETERS_TO_RANDOMIZE, prediction))

# join the predicted parameters with the overridden ones
base_patch = preset_to_patch(BASE_PATCH)
override_parameters, _ = split_train_override_patch(base_patch, PARAMETERS_TO_RANDOMIZE)

override_parameters.extend(pred)
override_parameters.sort()
full_predicted_patch = override_parameters

#make a preset from the predicted patch
preset = patch_to_preset(full_predicted_patch, os.path.join(DATA_PATH, f"predicted_patch.h2p"))


# EVALUATION ------------------------------------
# 1. AUDIO COMPARISON ---------------------------
# render the sound of the predicted patch
synth = spgl.synth.SynthVST(VST_PATH,
                            note_length_secs=2.0,
                            render_length_secs=2.0)

synth.set_patch(full_predicted_patch)
synth.get_patch()
predicted_render = synth.render_patch()
predicted_audio = synth.get_audio()
predicted_audio.save(os.path.join(AUDIO_PATH, f"predicted_eval_output_{eval_audio}.wav"))
print(f"The predicted audio file 'predicted_eval_output_{eval_audio}.wav' was saved in {AUDIO_PATH}")


# 2. COMPARISON PREDICTED PATCH VS TRUE PATCH
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
eval_patches = np.load(os.path.join(DATA_PATH, "eval_patches.npy"))

mse = tf.keras.metrics.mean_squared_error(eval_patches[eval_audio], prediction)
rmse = rmse(eval_patches[eval_audio], prediction)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")