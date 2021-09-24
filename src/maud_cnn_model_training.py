#!/usr/bin/env python
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from maud_conv_model import ConvModel

DATA_PATH = "../data/dataset_124params"
MODEL_PATH = "../models"
BATCH_SIZE = 64

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
dataset_train_original = tf.data.Dataset.from_tensor_slices((trainMels, train_target))
dataset_validate_original = tf.data.Dataset.from_tensor_slices((testMels, test_target))

# tf.data.experimental.save(dataset_train_original, os.path.join(DATA_PATH, "dataset_train_original"),
#                           compression=None, shard_func=None)
# tf.data.experimental.save(dataset_validate_original, os.path.join(DATA_PATH, "dataset_validate_original"),
#                           compression=None, shard_func=None)

# Prepare datasets for model training
dataset_train = dataset_train_original.cache().shuffle(10000).batch(BATCH_SIZE)
dataset_validate = dataset_validate_original.cache().batch(BATCH_SIZE) # No need to shuffle validation data


# Define custom loss function root mean squared error (MSE punishes big errors relatively more)
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.mean(tf.square(y_pred - y_true)))


# Define custom callback saving model's weights every n epochs
checkpoint_path = os.path.join(MODEL_PATH, "maud_training_24sep_1700/cp-{epoch:03d}.ckpt")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
    mode="auto",
    save_freq=2*math.ceil(trainMels.shape[0] / BATCH_SIZE)
)


# Train model
model = ConvModel(shape=(256, 347, 1),
                  output_size=124)

model.summary()

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Save model weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(
    dataset_train,
    epochs=3,
    validation_data=dataset_validate,
    callbacks=[cp_callback]
)

# Save the entire model as a SavedModel.
model.save(os.path.join(MODEL_PATH, "maud_training_24sep_1700/final_model"))


plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.show()
plt.close()


