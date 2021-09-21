#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from graham_lstm_model  import LstmHighwayModel

DATA_PATH = "../data/toy_data"
MODEL_PATH = "../models"
BATCH_SIZE = 128

# If model folder does not exist, create it.
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Load data files
trainMFCC = np.load(os.path.join(DATA_PATH, "train_0_features.npy"))
testMFCC = np.load(os.path.join(DATA_PATH, "test_0_features.npy"))

# Load training and validation targets
train_target = np.load(os.path.join(DATA_PATH, "train_0_patches.npy"))
test_target = np.load(os.path.join(DATA_PATH, "test_0_patches.npy"))

# Check dimensions of training and test data
print(f"The shape of trainMFCC: {trainMFCC.shape}")
print(f"The shape of trainTargets: {train_target.shape}")
print(f"The shape of testMFCC: {testMFCC.shape}")
print(f"The shape of testTargets: {test_target.shape}")

# Create tensorflow datasets
train_mfcc_original = tf.data.Dataset.from_tensor_slices((trainMFCC, train_target))
test_mfcc_original = tf.data.Dataset.from_tensor_slices((testMFCC, test_target))

# Prepare datasets for model training
mfcc_dataset_train = train_mfcc_original.cache().shuffle(10000).batch(BATCH_SIZE)
mfcc_dataset_validate = test_mfcc_original.cache().batch(BATCH_SIZE) # No need to shuffle validation data

# Train model
model = LstmHighwayModel(shape=(44, 13), num_outputs=4)
model.summary()

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

history = model.fit(
    mfcc_dataset_train,
    epochs=50,
    validation_data=mfcc_dataset_validate
)

# Save the entire model as a SavedModel.
model.save(os.path.join(MODEL_PATH, 'graham_lstm_model_21sep_1730'))


plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.show()
plt.close()


