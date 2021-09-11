import os
import json
import numpy as np
import spiegelib as spgl

# Set these paths accordingly (make sure data folder exists)
VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
DATA_PATH = "../data/testing/"

# Load synthesizer with spiegelib - adjust preferred length
synth = spgl.synth.SynthVST(VST_PATH,
                            note_length_secs=1.0,
                            render_length_secs=4.0)

# Generate initial random patch and save audio file
synth.randomize_patch()
patch = synth.get_patch()
synth.save_state(os.path.join(DATA_PATH, "patch_init.json"))
synth.render_patch()
audio = synth.get_audio()
audio.save(os.path.join(DATA_PATH, "audio_init.wav"), normalize=False)
synth.load_state(os.path.join(DATA_PATH, "patch_init.json"))
synth.render_patch()
synth.save_state(os.path.join(DATA_PATH, "patch_init_reloaded.json"))
audio = synth.get_audio()
audio.save(os.path.join(DATA_PATH, "audio_init_reloaded.wav"), normalize=False)


# synth.set_overridden_parameters(overridden_parameter)
# new_patch = synth.get_patch()


# Loop to programmatically set each parameter to zero once
for i in np.arange(len(patch)):

for i in np.arange(len(patch)):
    synth.load_state(os.path.join(DATA_PATH, "patch_init.json"))
    overridden_parameter = [(int(i), 0.0)]
    synth.set_patch(overridden_parameter)
    synth.save_state(os.path.join(DATA_PATH, f"patch_param{i}.json"))
    synth.render_patch()
    audio = synth.get_audio()
    audio.save(os.path.join(DATA_PATH, f"audio_param{i}.wav"), normalize=False)




synth.randomize_patch()
synth.render_patch()
sound_1 = synth.get_audio()
sound_1.save(os.path.join(DATA_PATH, "check_original.wav"))
synth.save_state(os.path.join(DATA_PATH,"sound_1_patch"))

for i in range(0,10):
    synth.load_synth_config(os.path.join(DATA_PATH,"sound_1_patch"))
    synth.render_patch()
    x = synth.get_audio()
    filename = "check2"+str(i)+".wav"
    x.save(os.path.join(DATA_PATH, filename))