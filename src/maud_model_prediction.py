#!/usr/bin/env python
import os
import tensorflow as tf
import spiegelib as spgl

from utils.patch_utils import *
from maud_feature_extractor import wav_to_mel

AUDIO_PATH = "../data/toy_eval_data/audio"
MODEL_PATH = "../models"
BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"
PARAMETERS_TO_RANDOMIZE = [86, 131, 148, 149]

#SOUND_TO_PREDICT_PARAMETERS_FROM = f"{PATH}predict_this.wav"



# Import sound for evaluation and convert to mel spectrogram
# !!! Write function that also detects silence and trims?
audio_mel = wav_to_mel(os.path.join(AUDIO_PATH, "eval_output_0.wav"))
print(audio_mel.shape)


# Load model
model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'maud_conv_model_18sep_1625'))

# prediction = model.predict(tf.expand_dims(mel, axis=0))[0] # add a dimension (for training samples) for prediction
#
#
#
#
#
#
#
# #join the predicted parameters with the overridden ones
# randomized_parameters = [86, 131, 148, 149]
# pred = list(zip(randomized_parameters, prediction))
#
# base_patch = preset_to_patch("../data/MS-REV1_deepdiva.h2p")
# _, override_parameters = split_train_override_patch(base_patch, randomized_parameters)
#
# override_parameters.extend(pred)
# override_parameters.sort()
# full_predicted_patch = override_parameters
#
# #make a preset from the predicted patch
# preset = patch_to_preset(full_predicted_patch, f"{PATH}/predicted_patch.h2p")
#
# #render the sound of the predicted patch with the renderman engine
# synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
#                             note_length_secs=1.0,
#                             render_length_secs=1.0)
#
# synth.set_patch(full_predicted_patch)
# synth.get_patch()
# predicted_render = synth.render_patch()
# predicted_audio = synth.get_audio()
# predicted_audio.save(f"{PATH}predicted_audio.wav")
