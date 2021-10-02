#!/usr/bin/env python
import os
import numpy as np
from tqdm import trange
import librenderman as rm
import scipy.io.wavfile
from deepdiva.utils.h2p_utils import H2P
from deepdiva.utils.patch_utils import split_train_override_patch


class DatasetGenerator():

    def __init__(self,
                 data_path,
                 vst_path,
                 folder_name,
                 base_preset,
                 random_parameters,
                 save_audio,
                 sample_rate,
                 midi_note_pitch,
                 midi_note_velocity,
                 note_length_seconds,
                 render_length_seconds):

        """
        Constructor
        """

        self.data_path = data_path
        self.vst_path = vst_path
        self.folder_name = folder_name
        self.base_preset = base_preset
        self.random_parameters = random_parameters
        self.save_audio = save_audio
        self.sample_rate = sample_rate
        self.midi_note_pitch = midi_note_pitch
        self.midi_note_velocity = midi_note_velocity
        self.note_length_seconds = note_length_seconds
        self.render_length_seconds = render_length_seconds


    def synth_configuration(self):
        engine = rm.RenderEngine(self.sample_rate, 512, 512)
        # Supply full path to the DIVA plugin to load the vst
        engine.load_plugin(self.vst_path)

        return engine


    def generate(self, sample_size, file_prefix):
        # Create relevant folders if not existing
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if not os.path.exists(os.path.join(self.data_path, self.folder_name)):
            os.makedirs(os.path.join(self.data_path, self.folder_name))

        if self.save_audio:
            if not os.path.exists(os.path.join(self.data_path, self.folder_name, "audio")):
                os.makedirs(os.path.join(self.data_path, self.folder_name, "audio"))

        # Set generator with synthesizer configuration
        engine = self.synth_configuration()
        generator = rm.PatchGenerator(engine)

        # Create one data example to infer size of arrays
        example_patch = generator.get_random_patch()
        engine.set_patch(example_patch)
        engine.render_patch(self.midi_note_pitch,
                            self.midi_note_velocity,
                            self.note_length_seconds,
                            self.render_length_seconds)
        example_audio = engine.get_audio_frames()

        # Get overridden parameters
        overridden_parameters = self.__get_overridden()
        overridden = [p[0] for p in overridden_parameters]

        # Assert number of data samples is specified
        assert sample_size is not None, "Please set the number of data samples required"

        # Initiate empty arrays to save data samples
        audio_set = np.zeros((sample_size, len(example_audio)), dtype=np.float32)
        patch_set = np.zeros((sample_size, len(self.random_parameters)), dtype=np.float32)

        for i in trange(sample_size):
            # Generate random patch
            random_patch = generator.get_random_patch()
            engine.set_patch(random_patch)

            # Overwrite fixed parameters and get full patch
            for param in overridden_parameters:
                engine.override_plugin_parameter(param[0], param[1])
            patch = engine.get_patch()

            # Render the patch
            engine.render_patch(self.midi_note_pitch,
                                self.midi_note_velocity,
                                self.note_length_seconds,
                                self.render_length_seconds)

            # Get decoded audio
            audio = engine.get_audio_frames()
            audio = np.array(audio, copy=True, dtype=np.float32)
            audio_set[i] = audio

            # Get synthesizer patch
            for element in sorted(overridden, reverse=True):
                patch.pop(element)
            patch_set[i] = [p[1] for p in patch]

            if self.save_audio:
                scipy.io.wavfile.write(
                    os.path.join(self.data_path, self.folder_name, "audio", f"{file_prefix}audio_{i}.wav"),
                    self.sample_rate,
                    audio)

        # Cut size audio array to exact length: sample_rate * render_length_seconds
        audio_set = audio_set[:, :int(self.sample_rate * self.render_length_seconds)]

        # Save dataset
        np.save(os.path.join(self.data_path, self.folder_name, f"{file_prefix}audio.npy"), audio_set)
        np.save(os.path.join(self.data_path, self.folder_name, f"{file_prefix}patches.npy"), patch_set)


    def __get_overridden(self):

        if self.base_preset == "MS-REV1_deepdiva.h2p":
            print("The default preset 'REV-MS1_deepdiva.h2p' is used for fixed parameters")

        assert os.path.exists(os.path.join(self.data_path, self.base_preset)), \
            f"The chosen preset {os.path.join(self.data_path, self.base_preset)} does not exist"

        used_base_preset = os.path.join(self.data_path, self.base_preset)
        h2p = H2P()
        base_patch = h2p.preset_to_patch(h2p_filename=used_base_preset)

        if self.random_parameters == []:
            print("None of the parameters is set to vary, all parameters will be fixed")

        overridden_parameters, _ = split_train_override_patch(base_patch, self.random_parameters)

        return overridden_parameters