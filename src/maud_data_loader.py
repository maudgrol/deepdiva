#!/usr/bin/env python
import os
import math
import numpy as np
import tensorflow as tf

TRAIN_PATH = "../data/dataset_124params/train_dataset"
VAL_PATH = "../data/dataset_124params/train_dataset"


class DataGenerator(tf.keras.utils.Sequence):
    """Generates dataset"""
    def __init__(self, list_ids, nr_params=124, batch_size=64, dim=(256, 347), n_channels=1, shuffle=True):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.nr_params = nr_params
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.list_ids) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.nr_params))

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            X[i, ] = np.load(os.path.join(TRAIN_PATH, f"train_melspectrogram_{ID}.npy"))

            # Store target
            y[i, ] = np.load(os.path.join(TRAIN_PATH, f"train_target_{ID}.npy"))

        return X, y
