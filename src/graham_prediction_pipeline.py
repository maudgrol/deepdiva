from tensorflow.keras import models
import utils.patch_utils as patch_utils
import spiegelib as spgl

PATH = "../small_data4/"
SOUND_TO_PREDICT_PARAMETERS_FROM = "../data/predict_this.wav"
PARAMETERS_TO_RANDOMIZE = [86, 131, 148, 149]

# Load the sound file
feature_extractor = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True)
feature_extractor.load_scaler(f"{PATH}data_scaler.pkl")

audio = spgl.AudioBuffer(SOUND_TO_PREDICT_PARAMETERS_FROM)
audio_mfcc = feature_extractor.get_features(audio)
audio_mfcc = feature_extractor.scale(audio_mfcc)

# Model prediction
model = models.load_model("../models/graham_lstm_model_55000_4parameters_22sep_1245")
prediction = model.predict(audio_mfcc)[0].astype("float64")

# join the predicted parameters with the overridden ones
randomized_parameters = [86, 131, 148, 149]
pred = list(zip(randomized_parameters, prediction))

base_patch = patch_utils.preset_to_patch("../data/MS-REV1_deepdiva.h2p")
_, override_parameters = patch_utils.split_train_override_patch(base_patch, randomized_parameters)

override_parameters.extend(pred)
override_parameters.sort()
full_predicted_patch = override_parameters

preset = patch_utils.patch_to_preset(full_predicted_patch, f"{PATH}/predicted_patch.h2p")



#####################

# BASE_PATCH = "../data/MS-REV1_deepdiva.h2p"
# PARAMETERS_TO_RANDOMIZE = [86, 131, 148, 149]
# SOUND_TO_PREDICT_PARAMETERS_FROM = f"{PATH}predict_this.wav"

# # import sound to predict and extract its features...
# features = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True)
# features.load_scaler(f"{PATH}data_scaler.pkl")

# audio = spgl.AudioBuffer(SOUND_TO_PREDICT_PARAMETERS_FROM)
# audio_mfcc = features.get_features(audio)
# audio_mfcc = features.scale(audio_mfcc)
# print(audio_mfcc.shape)

# #prepare shape for prediction
# audio_mfcc = np.expand_dims(audio_mfcc, axis=0)
# audio_mfcc = np.expand_dims(audio_mfcc, axis=-1)
# print(audio_mfcc.shape)

# #load the model and predict
# #apparently, if the prediction, which are originally float32, are not transformed into float64, they cannot render audio later....
# model = models.load_model(f"{PATH}model_linear")
# prediction = model.predict(audio_mfcc)[0].astype("float64")

# #join the predicted parameters with the overridden ones
# randomized_parameters = [86, 131, 148, 149]
# pred = list(zip(randomized_parameters, prediction))

# base_patch = preset_to_patch("../data/MS-REV1_deepdiva.h2p")
# _, override_parameters = split_train_override_patch(base_patch, randomized_parameters)

# override_parameters.extend(pred)
# override_parameters.sort()
# full_predicted_patch = override_parameters

# #make a preset from the predicted patch
# preset = patch_to_preset(full_predicted_patch, f"{PATH}/predicted_patch.h2p")

# #render the sound of the predicted patch with the renderman engine
# synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
#                             note_length_secs=1.0,
#                             render_length_secs=1.0)

# synth.set_patch(full_predicted_patch)
# synth.get_patch()
# predicted_render = synth.render_patch()
# predicted_audio = synth.get_audio()
# predicted_audio.save(f"{PATH}predicted_audio.wav")
