#!/usr/bin/env python
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from deepdiva.model.lstm_model import LstmHighwayModel
from deepdiva.model.cnn_model import ConvModel
from deepdiva.utils.model_utils import root_mean_squared_error


DATA_PATH = "../../../data/dataset_18params"
MODEL_PATH = "../../../models"
MODEL_FILE = "cnn_training_01oct"
BATCH_SIZE = 64
TRAIN_FEATURES = "train_melspectrogram.npy"
TEST_FEATURES = "test_melspectrogram.npy"
TRAIN_TARGET = "train_patches.npy"
TEST_TARGET = "train_patches.npy"
SAVE_WEIGHTS = True
SAVE_FREQ = 50
SAVE_MODEL = True
MODEL_TYPE = "cnn" #(or "lstm")
OPTIMIZER = "adam"
LOSS = "rmse"
METRICS = ["mean_absolute_error", "mean_squared_error"]
EPOCHS = 10




def main(model_type=MODEL_TYPE, data_path=DATA_PATH, model_path=MODEL_PATH, model_file=MODEL_FILE,
         batch_size=BATCH_SIZE, train_features=TRAIN_FEATURES, test_features=TEST_FEATURES,
         train_target=TRAIN_TARGET, test_target=TEST_TARGET, save_weights=SAVE_WEIGHTS, save_freq=SAVE_FREQ,
         save_model=SAVE_MODEL, optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS, epochs=EPOCHS):

    # If model folder does not exist, create it.
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Load training and validation features
    train_x = np.load(os.path.join(data_path, test_features))
    test_x = np.load(os.path.join(data_path, test_features))

    # Load training and validation targets
    train_y = np.load(os.path.join(data_path, train_target))
    test_y = np.load(os.path.join(data_path, test_target))

    # Infer input and number of outputs from data
    shape = train_x.shape[1:]
    num_outputs = train_y.shape[1]

    # Create tensorflow data sets
    dataset_train_original = tf.data.Dataset.from_tensor_slices((trainMels, train_target))
    dataset_validate_original = tf.data.Dataset.from_tensor_slices((testMels, test_target))

    # Prepare datasets for model training: shuffle and batch
    dataset_train = dataset_train_original.cache().shuffle(10000).batch(batch_size)
    dataset_validate = dataset_validate_original.cache().batch(batch_size) # No need to shuffle validation data

    # Define custom callback saving model weights every n epochs if True
    if save_weights:
        checkpoint_path = os.path.join(model_path, f"{model_file}", "cp-{epoch:03d}.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            mode="auto",
            save_freq=save_freq*math.ceil(len(train_x) / batch_size)
        )

    # Initiate model
    if model_type == "cnn":
        # Initiate convolutional neural network
        model = ConvModel(shape=shape, num_outputs=num_outputs)
        print("MODEL SUMMARY:")
        print(model.summary())


    if model_type == "lstm":
        # Initiate lstm model
        model = LstmHighwayModel(shape=shape, num_outputs=num_outputs)
        print("MODEL SUMMARY:")
        print(model.summary())

    # Compile the model
    if loss == "rmse":
        my_loss = root_mean_squared_error()

    model.compile(
        optimizer=optimizer,
        loss=my_loss,
        metrics=[metrics]
    )

    # Train the model
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_validate,
        callbacks=[cp_callback]
    )

    # Save final model if True
    if save_model:
        model.save(os.path.join(model_path, f"{model_file}/final_model"))

    # Plot training and validation loss after completion of model
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    plt.show()
    plt.close()
