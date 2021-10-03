import os
import sys
sys.path.append("/Users/grahamherdman/Documents/data-science-retreat/deep-diva/deepdiva/src")

from utils.patch_utils import split_train_override_patch, get_randomization_small, get_randomization_medium, get_randomization_big
from flask import Flask, request, send_from_directory, abort, Response
from tensorflow.keras import models 
from utils.h2p_utils import H2P
from flask_cors import CORS
import tensorflow as tf
from json import dumps
import numpy as np
import librosa
import soundfile

# configuration
DEBUG = True
AUDIO_LENGTH = 2

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# add h2p file to app.config
app.config["data_folder"] = "/Users/grahamherdman/Documents/data-science-retreat/deep-diva/deepdiva"

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# prediction route
@app.route('/prediction', methods=['POST'])
def prediction():
    # receive the audio file and save it in a temp file
    file = request.files['wavfile']

    os.system('rm audio.wav')
    with open("audio.wav", "wb") as audio_file:
        audio_stream = file.read()
        audio_file.write(audio_stream)

    start = int(request.form['start'])
    end = start + 2

    # check if the duration of the file is longer than end 
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
        n_fft=2048,
        win_length=2048,
        hop_length=1024,
        n_mfcc=13,
        sr=44100,
        fmin=50,
        fmax=15000
    )

    # add dimension for channel
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    # # flip frequency axis so low frequencies are at bottom of image
    # mfcc = mfcc[::-1, :, :]

    # scale mfcc
    min, max = np.load("/Users/grahamherdman/Documents/data-science-retreat/deep-diva/deepdiva/api/33para_200tsd__scaler.pckl", allow_pickle=True)
    scaled_mfcc = (mfcc - min) / (max - min)

    # load model
    model = models.load_model("/Users/grahamherdman/Documents/data-science-retreat/deep-diva/deepdiva/api/model33_200000_mfcc1387", compile=False)

    def root_mean_squared_error(y_true, y_pred):
        return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))

    my_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=my_optimizer,
        loss=root_mean_squared_error,
        metrics=["mean_squared_error"]
    )

    # prediction
    prediction = model.predict(scaled_mfcc)[0].astype("float64")

    # join the predicted parameters with the overridden ones
    PARAMETERS_TO_PREDICT = sorted(get_randomization_medium())
    pred = list(zip(PARAMETERS_TO_PREDICT, prediction))
    h2p = H2P()

    base_patch = h2p.preset_to_patch("/Users/grahamherdman/Documents/data-science-retreat/deep-diva/deepdiva/data/MS-Rev1_deepdiva.h2p")
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


    # scaler
    # mfcc
    # parameters to predict
    # model
