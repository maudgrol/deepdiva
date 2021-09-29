#!/usr/bin/env python
import click
import os
import numpy as np
from dotenv import find_dotenv, load_dotenv
from deepdiva.data.dataset_generator import DatasetGenerator


@click.command()
@click.option('--data-path', 'data_path', default="./data/", required=False,
              type=click.Path(), show_default=True, help='Path to data folder')
@click.option('--vst-path', 'vst_path', default="/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
              required=False, type=click.Path(exists=True), show_default=True, help='Path to vst plugin')
@click.option('--folder-name', 'folder_name', default="dataset",
              required=False, type=str, show_default=True, help='Folder name for saving dataset')
@click.option('--base-preset', 'base_preset', default="MS-REV1_deepdiva.h2p", required=False,
              type=str, show_default=True, help='DIVA preset that serves as base for fixed parameters')
@click.option('--random-parameters', 'random_parameters', type=str, default="",
              show_default=True, help="Indices of to be randomized parameters. Format: 'id id'")
@click.option('--save-audio/--no-save-audio', 'save_audio', default=False,
              show_default=True, help='Whether to save generated audio as .wav files')
@click.option('--sample-rate', 'sample_rate', default=44100, required=False,
              type=int, show_default=True, help='Sample rate for rendering audio')
@click.option('--midi-note-pitch', 'midi_note_pitch', default=48, required=False,
              type=int, show_default=True, help='Midi note (C3 is 48)')
@click.option('--midi-note-velocity', 'midi_note_velocity', default=127,
              required=False, type=int, show_default=True, help='Midi note velocity (0-127)')
@click.option('--note-length-seconds', 'note_length_seconds', default=2.0,
              required=False, type=float, show_default=True, help='Note length in seconds')
@click.option('--render-length-seconds', 'render_length_seconds', default=2.0,
              required=False, type=float, show_default=True, help='Rendered audio length in seconds')
@click.option('--sample-size', 'sample_size', required=True, type=int,
              show_default=True, help='Number of generated data samples per batch')
@click.option('--file-prefix', 'file_prefix', default="", required=False, type=str,
              show_default=True, help="Prefix for saving generated audio and patch files (e.g. 'train_')")
@click.option('--nr-data-batches', 'nr_batches', required=False, default=1, type=int,
              show_default=True, help='Number of data batches')
@click.option('--nr-batch-completed', 'nr_completed', required=False, default=0,
              type=int, show_default=True, help='Number of already generated data batches,')


def click_main(data_path, vst_path, folder_name, base_preset, random_parameters, save_audio,
               sample_rate, midi_note_pitch, midi_note_velocity, note_length_seconds,
               render_length_seconds, sample_size, file_prefix, nr_batches, nr_completed):
    """
    Interface for Click CLI.
    """
    # Create a list of integers of parameters to randomize
    random_parameters = random_parameters.split()
    random_parameters = [int(x) for x in random_parameters]

    # Generate dataset (in different batches)
    for batch in range(nr_completed, nr_batches):
        generator = DatasetGenerator(data_path=data_path, vst_path=vst_path, folder_name=folder_name,
                                     base_preset=base_preset, random_parameters=random_parameters,
                                     save_audio=save_audio, sample_rate=sample_rate,
                                     midi_note_pitch=midi_note_pitch, midi_note_velocity=midi_note_velocity,
                                     note_length_seconds=note_length_seconds, render_length_seconds=render_length_seconds)

        generator.generate(sample_size=sample_size, file_prefix=f"{file_prefix}{batch}_")

    # Concatenate batch files and save
    total_audio = np.concatenate([np.load(os.path.join(data_path, folder_name, f"{file_prefix}{batch}_audio.npy")) \
                                    for i in range(nr_batches)], axis=0)
    total_patches = np.concatenate([np.load(os.path.join(data_path, folder_name, f"{file_prefix}{batch}_patches.npy")) \
                                    for i in range(nr_batches)], axis=0)
    np.save(os.path.join(data_path, folder_name, f"{file_prefix}audio.npy"), total_audio)
    np.save(os.path.join(data_path, folder_name, f"{file_prefix}patches.npy"), total_patches)


if __name__ == '__main__':

    # Find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    click_main()

