# deepDIVA

In this project we aim to create a deep-learning syntheziser programmer using the vst plugin DIVA. We are creating a
tool that addresses the challenge of programming sound synthesizers. This requires a thorough technical understanding as
software synthesizers often have many parameters. To tackle this, we built a deep learning model for automatic
synthesizer programming that takes a target sound as input, predicts the parameter settings for the synthesizer that
cause it to emit as close a sound as possible, and generate a template that you can load in your DIVA synthesizer.

## Setup environment

It is advised to create a new Anaconda environment with python 3.7 for this project. Most of the required libraries can
be installed by running:

```angular2html
pip install -r requirements.txt
```

This project uses Spiegelib, a library for research and development related to Automatic Synthesizer Programming. One of
the dependencies for Spiegelib is the RenderMan library, which must be installed manually in order to enable
programmatic interaction with VST synthesizers. Instructions for installing RenderMan in a virtual environment are
provided [here](https://spiegelib.github.io/spiegelib/getting_started/installation.html#librenderman-conda-install).

Note: although these instructions advice to specify python 3.7 for the conda environment, it is also possible to create
an environment with python 3.9. Installation instructions for RenderMan then have to be adjusted accordingly.

## Making predictions with test data

To make a model prediction based on an input sound (.wav file) with our pre-trained model, run the following:

```angular2html
python src/deepdiva/model/make_prediction.py <input path>
<file name>
```

## Dataset for model training and validation

We generated and saved datasets for training and validating deep learning models. Additionally, we created a small audio
dataset for evaluation. These datasets can be downloaded from [ADD LINK] and saved in the data folder.

Diva has 281 parameters for controlling different operators and other global options that can be arranged in various
ways to create complex sounds. For our project we focused on a subset of 124 of these parameters and we have overridden
the other parameters using the settings taken form the standard DIVA MS-REV1 preset. For a more simplified experiment,
one could focus on a smaller subset of parameters.

The datasets were generated by setting the note length and render length to be two seconds. If one would like to capture
the release portion of a synth signal, the render length could be set to longer than the note length.

Data generation works by generating random patches from the synthesizer and saving the decoded audio and parameter
values in separate .npy files. The decoded audio files can then be used to extract features (see below).

To generate your own dataset with specific settings, please adjust src/deepdiva/data/make_dataset.py, and then run the
following:

```angular2html
python src/deepdiva/data/make_dataset.py
```

## Feature extraction

We extracted both mel-frequency cepstral coefficients (MFCC) and mel-scaled power spectrograms as features to experiment
with LSTM and Convolutional Neural Network architectures respectively.

### Mel-frequency cepstral coefficients

The number of MFCCs to return was set to 13, the length of the Fast Fourier Transform window was set to 2048, with the
number of samples between successive frames set to 1024. We set a time_major argument to True so that the orientation of
the output is (time_slices, features), which can be used in a LSTM model architecture. We normalized the MFCC dataset
based on the training dataset, rescaling the range of the data to [0,1]. The scaling information is saved in a pickle
file to be re-used on the validation dataset and on new data. The MFCC features are saved under '_mfcc.npy'.

### Mel-spectrogram

The number of Mel bands to generate was set to 128. The length of the Fast Fourier Transform window was set to 2048,
with the number of samples between successive frames set to 512. Each frame of audio was windowed by a Hann window of
length 2048. The lowest frequency and highest frequency (in Hz) were set to 0 and 20000 respectively. The mel-scaled
power spectrogram was then converted to decibel units and normalized. An extra dimension was added for channel so the
output is of shape (n_mels, time, channel), which can be used in a Convolutional Neural Network architecture. The mel
frequency bands axis was flipped so low frequencies are at the bottom of the created mel spectrogram. The mel-scaled
power spectrograms are saved under '_melspectrogram.npy'.

To extract features for your own dataset, run the following:

```angular2html
python src/deepdiva/data/feature_extraction.py
```

## Model training

To train our models based on your own dataset and extracted features, run the following:

```angular2html
python src/deepdiva/model/model_training.py
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
│   │   ├── data                  <- Scripts to generate data and extract features
│   │   │   └── make_dataset.py
│   │   │   └── feature_extraction.py
│   │   │
│   │   ├── models                <- Scripts to train models and use saved model to make predictions
│   │   │   └── make_prediction.py
│   │   │   └── train_model.py
│   │   │
│   │   ├── utils                 <- Scripts with helper functions
│   │   │   └── patch_utils.py
│   │   │   └── h2p_utils.py
│   │   │   └── norm_functions.py
│   │   │   └── scale_functions.py
│   │   │   └── dictionaries.py
└──
```


