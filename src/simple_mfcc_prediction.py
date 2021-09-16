from tensorflow.keras import models
import spiegelib as spgl
import numpy as np
from utils.patch_utils import *


#get a sound we want to use to predict its features...
features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                              hop_size=1024, time_major=True)

audio = spgl.AudioBuffer("../small_data3/predict_this.wav")
audio_mfcc= features.get_features(audio)
print(audio_mfcc.shape)

#prepare shape for prediction
audio_mfcc = np.expand_dims(audio_mfcc, axis=0)
audio_mfcc = np.expand_dims(audio_mfcc, axis=-1)
print(audio_mfcc.shape)

#load the model and predict
#apparently, if the prediction, which are originally float32, are not transformed into float64, they cannot render audio later....
model = models.load_model("../small_data3/model")
prediction = model.predict(audio_mfcc)[0].astype("float64")

#join the predicted parameters with the overridden ones
randomized_parameters = [86, 131, 148, 149]
pred = list(zip(randomized_parameters, prediction))

base_patch = preset_to_patch("../data/MS-REV1_deepdiva.h2p")
_, override_parameters = split_train_override_patch(base_patch, randomized_parameters)

override_parameters.extend(pred)
override_parameters.sort()
full_predicted_patch = override_parameters

print(f'Full predicted patch  {len(full_predicted_patch)}')
print(full_predicted_patch)
print(type(full_predicted_patch))

#make a preset from the predicted patch
preset = patch_to_preset(full_predicted_patch, "../small_data3/predicted_patch.h2p")

#render the predicted sound on the spot
synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
                            note_length_secs=1.0,
                            render_length_secs=1.0)

synth.set_patch(full_predicted_patch)
synth.get_patch()
predicted_render = synth.render_patch()
predicted_audio = synth.get_audio()
predicted_audio.save("../small_data3/predicted_audio.wav")

