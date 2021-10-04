# deepDIVA

In this project we aim to create a deep-learning syntheziser programmer using the vst plugin DIVA. We are creating a
tool that addresses the challenge of programming sound synthesizers. This requires a thorough technical understanding as
software synthesizers often have many parameters. To tackle this, we built a deep learning model for automatic
synthesizer programming that takes a target sound as input, predicts the parameter settings for the synthesizer that
cause it to emit as close a sound as possible, and generate a template that you can load in your DIVA synthesizer.

In future work, we aim to improve the sound matching performance of our deep neural network architecture. At this point in time the sound matching performance of our best performing architecture is too limited for an application that users can integrate into their workflow, at least not for a synthesizer as complex as the Diva. 


## Setup environment

It is advised to create a new Anaconda environment with python 3.7 for this project. Most of the required libraries can
be installed by running:

```angular2html
pip install -r requirements.txt
```

This project uses the RenderMan library, which must be installed manually in order to enable programmatic interaction
with VST synthesizers. Instructions for installing RenderMan in a virtual environment are
provided [here](https://spiegelib.github.io/spiegelib/getting_started/installation.html#librenderman-conda-install).

Note: although these instructions advice to specify python 3.7 for the conda environment, it is also possible to create
an environment with python 3.9. Installation instructions in step 5 and 6 then have to be adjusted accordingly.

## Making predictions with test data

To make a model prediction based on an input sound (.wav file) with our current model, run:

```angular2html
python src/deepdiva/model/make_prediction.py 
```

## Dataset for model training and validation

We generated and saved datasets for training and validating deep learning models. Additionally, we created a small audio
dataset for evaluation. These datasets can be downloaded from [ADD LINK] and saved in the data folder.

Diva has 281 parameters for controlling different operators and other global options that can be arranged in various
ways to create complex sounds. For our project we focused on a subset of 33 of these parameters (), and we have overridden
the other parameters using the settings taken from the standard DIVA MS-REV1 preset. For another experiment, one could focus on a different subset of parameters.

The datasets were generated by setting the note length and render length to be two seconds. If one would like to capture
the release portion of a synth signal, the render length could be set to longer than the note length.

Data generation works by generating random patches from the synthesizer and saving the decoded audio and parameter
values in separate .npy files. The decoded audio files can then be used to extract features (see below).

To generate your own dataset with specific settings, you can check out all the possible options you can pass when
running make_dataset.py:

```angular2html
python src/deepdiva/data/make_dataset.py --help

Usage: make_dataset.py [OPTIONS]

Interface for Click CLI.

Options:
--data-path PATH                Path to data folder  [default: ./data/]
--vst-path PATH                 Path to vst plugin  [default:
                                /Library/Audio/Plug-Ins/VST/u-he/Diva.vst]
--folder-name TEXT              Folder name for saving dataset  [default:
                                dataset]
--base-preset TEXT              DIVA preset that serves as base for fixed
                                parameters  [default: MS-REV1_deepdiva.h2p]
--random-parameters TEXT        Indices of to be randomized parameters.
                                Format: 'id id'  [default: ]
--save-audio / --no-save-audio  Whether to save generated audio as .wav
                                files  [default: no-save-audio]
--sample-rate INTEGER           Sample rate for rendering audio  [default:
                                44100]
--midi-note-pitch INTEGER       Midi note (C3 is 48)  [default: 48]
--midi-note-velocity INTEGER    Midi note velocity (0-127)  [default: 127]
--note-length-seconds FLOAT     Note length in seconds  [default: 2.0]
--render-length-seconds FLOAT   Rendered audio length in seconds  [default:
                                2.0]
--sample-size INTEGER           Number of generated data samples per batch
                                [required]
--file-prefix TEXT              Prefix for saving generated audio and patch
                                files (e.g. 'train_')  [default: ]
--nr-data-batches INTEGER       Number of data batches  [default: 1]
--nr-batch-completed INTEGER    Number of already generated data batches,
                                [default: 0]
--help                          Show this message and exit.
```

An example of making a small training dataset with only 4 parameters varying, and saving the audio .wav files:

```angular2html
python src/deepdiva/data/make_dataset.py --sample-size 10 --save-audio --folder-name "dataset_4params" --file-prefix "train_" --random-parameters '86 131 148 149'
```

## Feature extraction

We extracted both mel-frequency cepstral coefficients (MFCC) and mel-scaled spectrograms as features to experiment with
LSTM and Convolutional Neural Network architectures respectively.

### Mel-frequency cepstral coefficients

The number of MFCCs to return was set to 13, the length of the Fast Fourier Transform window was set to 2048, with the
number of samples between successive frames set to 1024. The lowest frequency and highest frequency (in Hz) were set to 50 and 15000 respectively. We set a time_major argument to True so that the orientation of
the output is (time_slices, features), which can be used in a LSTM model architecture. We normalized the MFCC dataset
based on the training dataset, rescaling the range of the data to [0,1]. The scaling information is saved in a pickle
file to be re-used on the validation dataset and on new data. The MFCC features are saved under 'mfcc.npy'.

### Mel-spectrogram

The number of Mel bands to generate was set to 128. The length of the Fast Fourier Transform window was set to 2048,
with the number of samples between successive frames set to 512. Each frame of audio was windowed by a Hann window of
length 2048. The lowest frequency and highest frequency (in Hz) were set to 50 and 15000 respectively. The mel-scaled
power spectrogram uses the decibel scale and was normalized to values ranging between [0,1]. An extra dimension was added for channel so the
output is of shape (n_mels, time, channels), which can be used in a Convolutional Neural Network architecture. The mel
frequency band axis was flipped so low frequencies are at the bottom of the created mel spectrogram. The mel-scaled
power spectrograms are saved under 'melspectrogram.npy'.

To extract features for your own generated dataset, you can check out all the possible settings you can pass when
running get_features.py:

```angular2html
python src/deepdiva/features/get_features.py --help

Usage: get_features.py [OPTIONS]

  Interface for Click CLI.

Options:
  --feature [spectrogram|mfcc]    Which type of feature to extract  [required]
  --data-path PATH                Path to dataset folder  [default:
                                  ./data/dataset]
  --data-file TEXT                Audio file (file.npy) from which to extract
                                  features  [required]
  --file-prefix TEXT              Prefix for saving generated feature files
                                  (e.g. 'train_')  [default: ]
  --saved-scaler / --no-saved-scaler
                                  Whether to use a previously saved data
                                  scaler object when extracting MFCCs
                                  [default: no-saved-scaler]
  --scaler-file TEXT              File name of saved data scaler object
  --scale-axis INTEGER            Axis or axes to use for calculating scaling
                                  parameteres. Defaults to 0, which scales
                                  each MFCC and time series component
                                  independently.  [default: 0]
  --n_fft INTEGER                 Length of the FFT window  [default: 2048]
  --win-length INTEGER            Each frame of audio is windowed by window()
                                  and will be of length win_length and then
                                  padded with zeros. Defaults to win_length =
                                  n_fft
  --hop_length INTEGER            Number of samples between successive frames
                                  [default: 512;required]
  --n_mels INTEGER                Number of Mel bands to generate  [default:
                                  128]
  --n_mfcc INTEGER                Number of MFCCs to return  [default: 13]
  --sample-rate INTEGER           Sampling rate of the incoming signal
                                  [default: 44100]
  --fmin INTEGER                  Lowest frequency (in Hz)  [default: 0]
  --fmax INTEGER                  Highest frequency (in Hz), defaults to
                                  sample rate // 2
  --time-major / --no-time-major  Change MFCC to shape (time_slices, n_mfcc)
                                  for modelling  [default: time-major]
  --help                          Show this message and exit.
```

An example of extracting normalized mel spectrogram features and setting some options:
```angular2html
python src/deepdiva/features/get_features.py --feature "spectrogram" --data-path "./data/dataset" --data-file "train_audio.npy" --file-prefix "train_" --fmax 20000
```

For mfcc features you have to indicate whether you want to use a previously saved scaler object for normalizing (e.g.
when extracting MFCCs for validation and test data). This defaults to False, so it will create, use and save a data
scaler object based on the data. An example of extracting mfcc features with a previously saved scaler object:
```angular2html
python src/deepdiva/features/get_features.py --feature "mfcc" --data-path "./data/dataset" --data-file "train_audio.npy" --file-prefix "test_" --fmax 20000 --saved-scaler --scaler-file "train_mfcc_scaling.pickle"
```

## Model training

To train one of our predefined models based on your own dataset and extracted features, you can check out all the possible settings you can pass when
running model_training.py:
```angular2html
python src/deepdiva/model/model_training.py --help

Usage: model_training.py [OPTIONS]

  Interface for Click CLI.

Options:
  --data-path PATH                Path to data folder  [default:
                                  ./data/dataset]
  --model-path PATH               Path to model folder  [default: ./models/]
  --folder-name TEXT              Folder name for saving model (weights)
                                  [default: training_<date>_<time>]
  --model [cnn|lstm]              Which type of pre-defined model to train
                                  [required]
  --train-features TEXT           File name of training features (.npy)
                                  [default: train_melspectrogram.npy]
  --test-features TEXT            File name of validation features (.npy)
                                  [default: test_melspectrogram.npy]
  --train-target TEXT             File name of training targets (.npy)
                                  [default: train_patches.npy]
  --test-target TEXT              File name of validations targets (.npy)
                                  [default: test_patches.npy]
  --batch_size INTEGER            Batch size  [default: 64]
  --epochs INTEGER                The number of complete passes through the
                                  training dataset  [default: 10]
  --save-model / --no-save-model  Whether to save final model  [default: save-
                                  model]
  --save-weights / --no-save-weights
                                  Whether to save model weights during
                                  training  [default: save-weights]
  --save_freq INTEGER             How often to save model weights (every n
                                  epochs)  [default: 50]
  --optimizer TEXT                Optimizer for model training (from
                                  tf.keras.optimizers)  [default: adam]
  --loss TEXT                     Loss function for model training  [default:
                                  rmse]
  --metrics TEXT                  Metrics to track during model training
                                  [default: mean_absolute_error]
  --help                          Show this message and exit.
```

An example of training a convolutional neural network model for 50 epochs, with the default train and test data file names, and saving the model weights every 10 epochs:
```angular2html
python src/deepdiva/model/model_training.py --data-path "./data/dataset" --model cnn --batch-size 64 --epochs 50 --save-freq 10 --loss "rmse" --metrics "mean_absolute_error"
```










## Project organization

```angular2html
├── LICENSE
├── README.md             <- The top-level README for developers using this project.
├── data                  <- Folder with saved dataset and pickle files.
├── models                <- Folder with saved models.
├── notebooks             <- Jupyter notebooks.
├── requirements.txt      <- The requirements file for reproducing the analysis environment
│
├── setup.py              <- makes project pip installable (pip install -e .) so deepdiva module can be imported
├── src                   <- Source code for use in this project.
│   ├── deepdiva
│   │   ├── __init__.py           <- Makes src/deepdiva a Python module
│   │   │
│   │   ├── data                  <- Scripts to generate data
│   │   │   └── make_dataset.py
│   │   │   └── dataset_generator.py
│   │   │   └── data_inspection.py
│   │   │
│   │   ├── features              <- Scripts to extract features
│   │   │   └── get_features.py
│   │   │   └── feature_extractor.py
│   │   │
│   │   ├── models                <- Scripts to train models and use saved model to make predictions
│   │   │   └── cnn_model.py
│   │   │   └── lstm_model.py
│   │   │   └── highway_layer.py
│   │   │   └── model_prediction.py
│   │   │   └── model_training.py
│   │   │
│   │   ├── utils                 <- Scripts with helper functions
│   │   │   └── patch_utils.py
│   │   │   └── h2p_utils.py
│   │   │   └── norm_functions.py
│   │   │   └── scale_functions.py
│   │   │   └── dictionaries.py
│   │   │   └── visualisation_utils.py
│   │   │   └── model_utils.py
└──
```


