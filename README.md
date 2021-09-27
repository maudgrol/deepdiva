# deepDIVA

In this project we aim to create a deep-learning syntheziser programmer using the vst plugin DIVA. We are creating a
tool that addresses the challenge of programming sound synthesizers. This requires a thorough technical understanding as
software synthesizers often have many parameters. To tackle this, we built a deep learning model for automatic
synthesizer programming that takes a target sound as input, predicts the parameter settings for the synthesizer that
cause it to emit as close a sound as possible, and generate a template that you can load in your DIVA synthesizer.

## Setup environment

It is advised to create a new Anaconda environment with python 3.7 for this project. Most of the required libraries can
be installed by running:

```
pip install -r requirements.txt
```

This project uses Spiegelib, a library for research and development related to Automatic Synthesizer Programming. One of
the dependencies for Spiegelib is the RenderMan library, which must be installed manually in order to enable
programmatic interaction with VST synthesizers. Instructions for installing RenderMan in a virtual environment are
provided [here](https://spiegelib.github.io/spiegelib/getting_started/installation.html#librenderman-conda-install).

Note: although these instructions advice to specify python 3.7 for the conda environment, it is also possible to create
an environment with python 3.9. Installation instructions for RenderMan then have to be adjusted accordingly.

## Dataset

We generated and saved datasets for training and validation of our deep learning models. 

Here we generate and save datasets for training and validating deep learning models. Additionally, we create a small
audio dataset for evaluation.

import spiegelib as spgl Load Dexed and set the note length and render length to be one second. For this experiment we
aren’t worried about the release of the sound. If we wanted to capture the release portion of a synth signal, we could
set the render length to longer than the note length. We’ll also reload the configuration JSON file previously saved.

