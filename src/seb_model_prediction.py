#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import librosa
import soundfile
import spiegelib as spgl
import pandas as pd
from utils.patch_utils import preset_to_patch, patch_to_preset, split_train_override_patch, get_randomization_small
import plotly.express as px

#instantiate the model
class ConvModel(tf.keras.Model):

    def __init__(self, shape, num_outputs, **kwargs):
        super(ConvModel, self).__init__()
        self.shape = shape

        # Define all layers
        # Layer of convolutional block 1
        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   activation="relu",
                                   name="Conv_1")
        self.max1 = layers.MaxPooling2D(pool_size=(2, 2),
                                        name="MaxPool_1")

        # Layer of convolutional block 2
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   activation="relu",
                                   name="Conv_2")
        self.max2 = layers.MaxPooling2D(pool_size=(2, 2),
                                        name="MaxPool_2")

        # Layer of convolutional block 3
        self.conv3 = layers.Conv2D(filters=128,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   activation="relu",
                                   name="Conv_3")
        self.max3 = layers.MaxPooling2D(pool_size=(2, 2),
                                        name="MaxPool_3")

        # Layer of convolutional block 4
        self.conv4 = layers.Conv2D(filters=128,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   activation="relu",
                                   name="Conv_4")
        self.max4 = layers.MaxPooling2D(pool_size=(2, 2),
                                        name="MaxPool_4")

        # Fully connected layers and dropout
        self.dropout1 = layers.Dropout(rate=0.2,
                                       name="Dropout_1")
        self.flatten = layers.Flatten(name="Flatten_1")
        self.fc1 = layers.Dense(units=128,
                                activation="relu",
                                name="Dense_1")
        self.dropout2 = layers.Dropout(rate=0.4,
                                       name="Dropout_2")
        self.fc2 = layers.Dense(units=num_outputs,
                                activation="linear",
                                name="Output_layer")

        self._build_graph()


    def _build_graph(self):
        self.build((None,) + self.shape)
        inputs = tf.keras.Input(shape=(self.shape))
        self.call(inputs)


    def call(self, input_tensor, training=None):
        # forward pass: convolutional block 1
        x = self.conv1(input_tensor)
        x = self.max1(x)

        # forward pass: convolutional block 2
        x = self.conv2(x)
        x = self.max2(x)

        # forward pass: convolutional block 3
        x = self.conv3(x)
        x = self.max3(x)

        # forward pass: convolutional block 4
        x = self.conv4(x)
        x = self.max4(x)

        # forward pass: dense layers, dropout and output
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x, training=training)
        return self.fc2(x)

DATA_PATH = "../data/to_predict"
AUDIO_PATH = "../data/to_predict/audio"
MODEL_PATH = "../models"
VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"

PARAMETERS_TO_RANDOMIZE = get_randomization_small() #18 parameters

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
        fmin=50,
        fmax=15000,
        center=True,
        power=2.0
    )

    # convert to decibel-scale melspectrogram: compute dB relative to peak power
    spectrogram = librosa.core.power_to_db(spectrogram, ref=np.max)
    # normalize data
    spectrogram = (spectrogram +80) / 80
    # add dimension for channel
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    # flip frequency axis so low frequencies are at bottom of image
    spectrogram = spectrogram[::-1, :, :]

    return spectrogram

# load model
# model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "maud_training_26sep_08", "final_model_124params"))
model = ConvModel(shape=(128, 174, 1),
                      num_outputs=18)

# latest = tf.train.latest_checkpoint(os.path.join(MODEL_PATH, "seb"))
model.load_weights(os.path.join(MODEL_PATH, "seb/cp-050.ckpt"))

#make an empty pandas df with columns as parameter names
plist = []
[plist.append(str(i)) for i in PARAMETERS_TO_RANDOMIZE]
pred_errors = pd.DataFrame(columns=plist)

#loop over predictions and put errors into the df
for i in range(50):
    eval_mel = lib_wav_to_mel(os.path.join(AUDIO_PATH, f"eval_0_audio_{i}.wav"))
    #print(eval_mel.shape)

    # add a dimension (for training samples) for prediction
    y_predict = model.predict(tf.expand_dims(eval_mel, axis=0))[0].astype("float64")

    true_patches = np.load(os.path.join(DATA_PATH, "eval_patches.npy"))

    y_true = true_patches[i]

    error = y_predict - y_true
    error = np.expand_dims(error, axis=-1)
    error = np.transpose(error)
    error = pd.DataFrame(error, columns=plist)
    pred_errors = pd.concat([pred_errors, error], axis=0)

    # use the predicted values for trainable parameters
    pred = list(zip(PARAMETERS_TO_RANDOMIZE, y_predict))

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
    predicted_audio.save(os.path.join(AUDIO_PATH, f"predicted_{i}.wav"))





pred_errors.reset_index(inplace=True)


fig = px.box(pred_errors, y=plist, points="all") #hover_data=[pred_errors.index]
fig.show()

pred_transpose = pred_errors.transpose()

fig = px.box(pred_transpose, y=pred_transpose.columns, points="all") #hover_data=[pred_errors.index]
fig.show()




