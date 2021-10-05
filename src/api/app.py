# Mini11 

# RUN ME FROM API FOLDER
from utils.patch_utils import split_train_override_patch, get_randomization_small, get_randomization_medium, get_randomization_big
from flask import Flask, request, send_from_directory, abort, Response
from tensorflow.keras import models 
from tensorflow.keras import layers
from utils.h2p_utils import H2P
from flask_cors import CORS
from json import dumps
import tensorflow as tf
import numpy as np
import soundfile
import librosa
import os

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# configuration 
app.config["data_folder"] = "./"
AUDIO_LENGTH = 2
SCALER_FILE_PATH = "./33para_200tsd__scaler.pckl"
MODEL_WEIGHTS_FILE_PATH = "./mini11_maud/cp-0075.ckpt"
BASE_PATCH_PATH = "../../data/MS-Rev1_deepdiva.h2p"
MELSPECTROGRAM_PARAMETERS = {
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "sr": 44100,
    "fmin": 50,
    "fmax": 15000,
    "center": True,
    "power": 2.0
}
PARAMETERS_TO_PREDICT = [33, 34, 35, 86, 87, 97, 98, 131, 132, 148, 149]

# helper functions
def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))

def save_audio_file(file):
    os.system('rm audio.wav')
    with open("audio.wav", "wb") as audio_file:
        audio_stream = file.read()
        audio_file.write(audio_stream)

def get_spectrogram(audio, MELSPECTROGRAM_PARAMETERS):

    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        n_fft=MELSPECTROGRAM_PARAMETERS["n_fft"],
        win_length=MELSPECTROGRAM_PARAMETERS["win_length"],
        hop_length=MELSPECTROGRAM_PARAMETERS["hop_length"],
        n_mels=MELSPECTROGRAM_PARAMETERS["n_mels"],
        sr=MELSPECTROGRAM_PARAMETERS["sr"],
        fmin=MELSPECTROGRAM_PARAMETERS["fmin"],
        fmax=MELSPECTROGRAM_PARAMETERS["fmax"],
        center=MELSPECTROGRAM_PARAMETERS["center"],
        power=MELSPECTROGRAM_PARAMETERS["power"]
    )

    # convert to decibel-scale melspectrogram: compute dB relative to peak power
    spectrogram = librosa.core.power_to_db(spectrogram, ref=np.max)
    # normalize data
    spectrogram = (spectrogram - (-80)) / (0 - (-80))
    # add dimension for channel
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    # spectrogram = np.expand_dims(spectrogram, axis=0)
    # flip frequency axis so low frequencies are at bottom of image
    spectrogram = spectrogram[::-1, :, :]

    return spectrogram

# Convolutional neural network
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
        self.flatten = layers.Flatten(name="Flatten_1")
        self.fc1 = layers.Dense(units=128,
                                activation="relu",
                                name="Dense_1")
        self.dropout1 = layers.Dropout(rate=0.3,
                                       name="Dropout_1")
        self.fc2 = layers.Dense(units=64,
                                activation="relu",
                                name="Dense_2")
        self.dropout2 = layers.Dropout(rate=0.3,
                                       name="Dropout_2")
        self.fc3 = layers.Dense(units=num_outputs,
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
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return self.fc3(x)

# General #####################
def create_preset(PARAMETERS_TO_PREDICT, prediction, BASE_PATCH_PATH):
    # join the predicted parameters with the overridden ones
    pred = list(zip(PARAMETERS_TO_PREDICT, prediction))
    h2p = H2P()

    base_patch = h2p.preset_to_patch(BASE_PATCH_PATH)
    override_parameters, _ = split_train_override_patch(base_patch, PARAMETERS_TO_PREDICT)

    override_parameters.extend(pred)
    override_parameters.sort()
    full_predicted_patch = override_parameters

    # make a preset from the predicted patch
    predicted_patch_filename = "predicted_patch.h2p"
    h2p.patch_to_preset(full_predicted_patch, predicted_patch_filename)

    return predicted_patch_filename

def get_prediction(MODEL_WEIGHTS_FILE_PATH, spectrogram):
    import pdb; pdb.set_trace();

    model = ConvModel(shape=(128, 173, 1), num_outputs=11)

    #latest = tf.train.latest_checkpoint(os.path.join(MODEL_PATH, MODEL_FOLDER))
    model.load_weights(MODEL_WEIGHTS_FILE_PATH)

    # add a dimension (for training samples) for prediction
    prediction = model.predict(tf.expand_dims(spectrogram, axis=0))[0].astype("float64")

    return prediction
##########

# prediction route
@app.route('/prediction', methods=['POST'])
def prediction():

    file = request.files['wavfile']

    save_audio_file(file)

    start = int(request.form['start'])
    end = start + 2

    # check if the duration of the file is longer than the end 
    duration = librosa.get_duration(filename='audio.wav')
    if end > duration:
        error_message = dumps({
            'Message': 'The resulting sound is too short. Try an earlier starting point.'
        })
        abort(Response(error_message, 401))
    
    # trim the audio file
    os.system('rm trimmed_audio.wav')
    os.system(f'ffmpeg -ss {start} -t {AUDIO_LENGTH} -i audio.wav trimmed_audio.wav')

    # feature extraction - mfccsx
    audio_binary = tf.io.read_file("trimmed_audio.wav")
    try:
      audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=-1)
    except:
      data, samplerate = soundfile.read(file)
      soundfile.write(file, data, samplerate, subtype='PCM_16')
      audio_binary = tf.io.read_file(file)
      audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=-1)
    audio = audio[:, 0].numpy().astype("float32")

    # get spectrogram or mfcc
    spectrogram = get_spectrogram(audio, MELSPECTROGRAM_PARAMETERS)

    prediction = get_prediction(MODEL_WEIGHTS_FILE_PATH, spectrogram)

    predicted_patch_filename = create_preset(PARAMETERS_TO_PREDICT, prediction, BASE_PATCH_PATH)

    try:
        return send_from_directory(app.config["data_folder"], filename=predicted_patch_filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    app.run()
