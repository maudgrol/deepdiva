import os
import json
import spiegelib as spgl
import librenderman as rm

VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
PARAMS_PATH = "../synth_params/"
DATA_PATH = "../data/"

# If synthesizer parameter folder does not exist, create it.
if not os.path.exists(PARAMS_PATH):
    os.makedirs(PARAMS_PATH)

# Get synthesizer parameters with RenderMan package
engine = rm.RenderEngine(44100, 512, 512)
if engine.load_plugin(VST_PATH):
    print("Synthesizer loaded succesfully")

description = engine.get_plugin_parameters_description().splitlines()

# Load synthesizer with spiegelib
synth = spgl.synth.SynthVST(VST_PATH,
                            sample_rate=44100,
                            buffer_size=512,
                            midi_note=48,
                            midi_velocity=127,
                            note_length_secs=1.0,
                            render_length_secs=2.0,
                            overridden_params=None,
                            clamp_params=True)


# Get possible parameters
param_list = synth.get_parameters()

# Save initial state of synthesizer
synth.save_state(os.path.join(PARAMS_PATH, "diva_init.json"))

with open(os.path.join(PARAMS_PATH, "diva_init.json")) as f:
  diva_dict = json.load(f)

synth.load_synth_config(os.path.join(PARAMS_PATH, "diva_init.json"))


# render an audio file
audio = synth.get_random_example()
audio.save(os.path.join(DATA_PATH, "audio_init.wav"), normalize=False)
