#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from maud_conv_model import ConvModel


DATA_PATH = "../data/toy_data"
MODEL_PATH = "../models"
BATCH_SIZE = 128

# If model folder does not exist, create it.
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Load data files
trainMels = np.load(os.path.join(DATA_PATH, "train_melspectrogram.npy"))
testMels = np.load(os.path.join(DATA_PATH, "test_melspectrogram.npy"))

# Load training and validation targets
train_target = np.load(os.path.join(DATA_PATH, "train_patches.npy"))
test_target = np.load(os.path.join(DATA_PATH, "test_patches.npy"))

# Check dimensions of training and test data
print(f"The shape of trainMels: {trainMels.shape}")
print(f"The shape of trainTargets: {train_target.shape}")
print(f"The shape of testMels: {testMels.shape}")
print(f"The shape of testTargets: {test_target.shape}")

# Create tensorflow datasets
train_mel_original = tf.data.Dataset.from_tensor_slices((trainMels, train_target))
test_mel_original = tf.data.Dataset.from_tensor_slices((testMels, test_target))

# Save datasets - but seems tricky to reload because you need this spec variable saved
filename = os.path.join(DATA_PATH, "train_dataset_original")
tf.data.experimental.save(train_mel_original, filename, compression=None, shard_func=None)

filename = os.path.join(DATA_PATH, "test_dataset_original")
tf.data.experimental.save(test_mel_original, filename, compression=None, shard_func=None)

spec = train_mel_original.element_spec


# Prepare datasets for model training
mel_dataset_train = train_mel_original.cache().shuffle(10000).batch(BATCH_SIZE)
mel_dataset_validate = test_mel_original.cache().batch(BATCH_SIZE) # No need to shuffle validation data
mfcc_dataset_train = train_mfcc_original.cache().shuffle(10000).batch(BATCH_SIZE)
mfcc_dataset_validate = test_mfcc_original.cache().batch(BATCH_SIZE) # No need to shuffle validation data

# Train model
model = ConvModel(input_size=64, output_size=4)
model.summary()

model.compile(
    optimizer="adam",
    loss="cosine_similarity"
)

history = model.fit(
    mel_dataset_train,
    epochs=50,
    validation_data=mel_dataset_validate
)

# Save the entire model as a SavedModel.
model.save(os.path.join(MODEL_PATH, 'maud_conv_model_18sep_1625'))


plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.show()
plt.close()


