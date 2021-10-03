# RUN ME FROM API FOLDER

from utils.patch_utils import split_train_override_patch, get_randomization_small, get_randomization_medium, get_randomization_big
from flask import Flask, request, send_from_directory, abort, Response
from tensorflow.keras import models 
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
MODEL_FILE_PATH = "./model33_200000_mfcc1387"
BASE_PATCH_PATH = "../../data/MS-Rev1_deepdiva.h2p"
MFCC_PARAMETERS = {
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 1024,
    "n_mfcc": 13,
    "sr": 44100,
    "fmin": 50,
    "fmax": 15000,
}
PARAMETERS_TO_PREDICT = sorted(get_randomization_medium())

# helper functions
def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))

def save_audio_file(request):
    file = request.files['wavfile']

    os.system('rm audio.wav')
    with open("audio.wav", "wb") as audio_file:
        audio_stream = file.read()
        audio_file.write(audio_stream)

# prediction route
@app.route('/prediction', methods=['POST'])
def prediction():

    save_audio_file(request)

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

    mfcc = librosa.feature.mfcc(
        y=audio,
        n_fft=MFCC_PARAMETERS["n_fft"],
        win_length=MFCC_PARAMETERS["win_length"],
        hop_length=MFCC_PARAMETERS["hop_length"],
        n_mfcc=MFCC_PARAMETERS["n_mfcc"],
        sr=MFCC_PARAMETERS["sr"],
        fmin=MFCC_PARAMETERS["fmin"],
        fmax=MFCC_PARAMETERS["fmax"]
    )

    # add dimension for channel
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)

    # scale mfcc
    min, max = np.load(SCALER_FILE_PATH, allow_pickle=True)
    scaled_mfcc = (mfcc - min) / (max - min)

    # load model
    model = models.load_model(MODEL_FILE_PATH, compile=False)

    my_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=my_optimizer,
        loss=root_mean_squared_error,
        metrics=["mean_squared_error"]
    )

    # prediction
    prediction = model.predict(scaled_mfcc)[0].astype("float64")

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

    try:
        return send_from_directory(app.config["data_folder"], filename=predicted_patch_filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    app.run()
