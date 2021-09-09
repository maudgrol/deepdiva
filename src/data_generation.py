import spiegelib as spgl
import os

# Path variables
VST_PATH = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
PARAMS_PATH = "../synth_params/"
DATA_PATH = "../data/"

# Data generator variables
N_TRAIN = 100
N_TEST = 20
N_EVAL = 5

# If output folder for data does not exist, create it.
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Load synthesizer
synth = spgl.synth.SynthVST(VST_PATH,
                            note_length_secs=1.0,
                            render_length_secs=1.0)

# Load state synthesizer
synth.load_state(os.path.join(PARAMS_PATH, "diva_init.json"))

# Mel-frequency Cepstral Coefficients audio feature extractor.
features_mfcc = spgl.features.MFCC(num_mfccs=13,
                                   frame_size=2048,
                                   hop_size=1024,
                                   time_major=True)

# Setup data generator for MFCC output and generate training and test examples.
generator_mfcc = spgl.DatasetGenerator(synth, features_mfcc,
                                       output_folder=os.path.join(DATA_PATH, "data_mfcc"),
                                       scale=True,
                                       save_audio=False)
generator_mfcc.generate(N_TRAIN, file_prefix="train_")
generator_mfcc.generate(N_TEST, file_prefix="test_")
generator_mfcc.save_scaler("mfcc_data_scaler.pkl")

# Short Time Fourier Transform audio feature extractor.
features_stft = spgl.features.STFT(fft_size=512,
                                   hop_size=256,
                                   output="magnitude",
                                   time_major=True)

# Setup data generator for STFT output and generate training and test examples.
generator_stft = spgl.DatasetGenerator(synth, features_stft,
                                       output_folder=os.path.join(DATA_PATH, "data_stft"),
                                       scale=True,
                                       save_audio=False)
generator_stft.generate(N_TRAIN, file_prefix="train_")
generator_stft.generate(N_TEST, file_prefix="test_")
generator_stft.save_scaler("stft_data_scaler.pkl")

# Setup data generator for evaluation data audio files
generator_eval = spgl.DatasetGenerator(synth, features_mfcc,
                                       output_folder=os.path.join(DATA_PATH, "evaluation"),
                                       save_audio=True)
generator_eval.generate(N_EVAL, file_prefix="eval_")