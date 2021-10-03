#!/usr/bin/env python
import click
import sys
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from deepdiva.utils.visualisation_utils import plot_loss
from deepdiva.model.lstm_model import LstmHighwayModel
from deepdiva.model.cnn_model import ConvModel
from deepdiva.utils.model_utils import root_mean_squared_error

date = datetime.now()
date = date.strftime("%d-%m-%y_%H:%M:%S")

@click.command()
@click.option('--data-path', 'data_path', default="./data/dataset", required=False,
              type=click.Path(), show_default=True, help='Path to data folder')
@click.option('--model-path', 'model_path', default="./models/",
              required=False, type=click.Path(), show_default=True, help='Path to model folder')
@click.option('--folder-name', 'folder_name', default=f"training_{date}",
              required=False, type=str, show_default=True, help='Folder name for saving model (weights)')
@click.option('--model', 'model_type', type=click.Choice(['cnn', 'lstm'], case_sensitive=False),
              required=True, show_default=True, help="Which type of pre-defined model to train")
@click.option('--train-features', 'train_features', default="train_melspectrogram.npy",
              required=False, type=str, show_default=True, help='File name of training features (.npy)')
@click.option('--test-features', 'test_features', default="test_melspectrogram.npy",
              required=False, type=str, show_default=True, help='File name of validation features (.npy)')
@click.option('--train-target', 'train_target', default="train_patches.npy",
              required=False, type=str, show_default=True, help='File name of training targets (.npy)')
@click.option('--test-target', 'test_target', default="test_patches.npy",
              required=False, type=str, show_default=True, help='File name of validations targets (.npy)')
@click.option('--batch_size', 'batch_size', default=64, required=False,
              type=int, show_default=True, help='Batch size')
@click.option('--epochs', 'epochs', default=10, required=False,
              type=int, show_default=True, help='The number of complete passes through the training dataset')
@click.option('--save-model/--no-save-model', 'save_model', default=True,
              show_default=True, help='Whether to save final model')
@click.option('--save-weights/--no-save-weights', 'save_weights', default=True,
              show_default=True, help='Whether to save model weights during training')
@click.option('--save_freq', 'save_freq', default=50, required=False,
              type=int, show_default=True, help='How often to save model weights (every n epochs)')
@click.option('--optimizer', 'optimizer', default="adam",
              required=False, type=str, show_default=True, help='Optimizer for model training (from tf.keras.optimizers)')
@click.option('--loss', 'loss', default="rmse",
              required=False, type=str, show_default=True, help='Loss function for model training')
@click.option('--metrics', 'metrics', default=["mean_absolute_error"], multiple=True,
              required=False, type=str, show_default=True, help='Metrics to track during model training')


def click_main(data_path, model_path, folder_name, model_type, train_features, test_features,
               train_target, test_target, batch_size, epochs, save_model, save_weights, save_freq,
               optimizer, loss, metrics):
    """
    Interface for Click CLI.
    """

    main(data_path=data_path, model_path=model_path, folder_name=folder_name, model_type=model_type,
         train_features=train_features, test_features=test_features, train_target=train_target,
         test_target=test_target, batch_size=batch_size, epochs=epochs, save_model=save_model,
         save_weights=save_weights, save_freq=save_freq, optimizer=optimizer, loss=loss, metrics=metrics)


def main(data_path, model_path, folder_name, model_type, train_features, test_features, train_target, test_target,
         batch_size, epochs, save_model, save_weights, save_freq, optimizer, loss, metrics):
    """Runs model training script with predefined model architectures"""

    # If model folder does not exist, create it.
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Check that save frequency is not larger than number of epochs
    if epochs < save_freq:
        print("The frequency of saving model weights is larger than the total number of epochs.")
        value = input("Do you want to continue anyway? y/n: ")
        if value == "n":
            sys.exit(0)

    # Create tensorflow data sets
    dataset_train_original = tf.data.Dataset.from_tensor_slices(
        (np.load(os.path.join(data_path, train_features)), np.load(os.path.join(data_path, train_target))))
    dataset_validate_original = tf.data.Dataset.from_tensor_slices(
        (np.load(os.path.join(data_path, test_features)), np.load(os.path.join(data_path, test_target))))

    # Prepare datasets for model training: shuffle and batch
    dataset_train = dataset_train_original.cache().shuffle(10000).batch(batch_size)
    dataset_validate = dataset_validate_original.cache().batch(batch_size)

    # Define custom callback saving model weights every n epochs if True
    if save_weights:
        checkpoint_path = os.path.join(model_path, f"{folder_name}", "cp-{epoch:04d}.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            mode="auto",
            save_freq=save_freq*math.ceil(len(dataset_train_original) / batch_size)
        )

    # Infer input shape and number of outputs from data
    shape = dataset_train_original.element_spec[0]._shape_tuple
    num_outputs = dataset_train_original.element_spec[1].shape.dims[0].value

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
        my_loss = root_mean_squared_error
    else:
        my_loss = loss

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
        model.save(os.path.join(model_path, f"{folder_name}/final_model"))

    # Plot training and validation loss after completion of model
    plot_loss(history)


if __name__ == '__main__':

    click_main()
