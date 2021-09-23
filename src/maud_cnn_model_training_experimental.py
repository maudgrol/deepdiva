#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from maud_conv_model_custom_layer_experimental import ConvModel


DATA_PATH = "../data/dataset_4params"
MODEL_PATH = "../models"
BATCH_SIZE = 128

# If model folder does not exist, create it.
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Load data files
trainAudio = np.load(os.path.join(DATA_PATH, "train_audio_decoded.npy"))
testAudio = np.load(os.path.join(DATA_PATH, "test_audio_decoded.npy"))

# Load training and validation targets
train_target = np.load(os.path.join(DATA_PATH, "train_patches.npy"))
test_target = np.load(os.path.join(DATA_PATH, "test_patches.npy"))

# Check dimensions of training and test data
print(f"The shape of trainAudio: {trainAudio.shape}")
print(f"The shape of trainTargets: {train_target.shape}")
print(f"The shape of testAudio: {testAudio.shape}")
print(f"The shape of testTargets: {test_target.shape}")

# Create tensorflow datasets
dataset_train_original = tf.data.Dataset.from_tensor_slices((trainAudio, train_target))
dataset_validate_original = tf.data.Dataset.from_tensor_slices((testAudio, test_target))

# Prepare datasets for model training
dataset_train = dataset_train_original.cache().shuffle(10000).batch(BATCH_SIZE)
dataset_validate = dataset_validate_original.cache().batch(BATCH_SIZE) # No need to shuffle validation data

# define custom loss function root mean squared error
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.mean(tf.square(y_pred - y_true)))


# Train model
model = ConvModel(input_size=88576,
                  output_size=4,
                  sample_rate=44100,
                  fft_size=4096,
                  hop_size=256,
                  n_mels=256,
                  f_min=0,
                  f_max=20000)

model.summary()

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

history = model.fit(
    dataset_train,
    epochs=2,
    validation_data=dataset_validate
)

# # # Save the entire model as a SavedModel.
# # model.save(os.path.join(MODEL_PATH, 'maud_conv_model_18sep_1625'))


plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.show()
plt.close()


