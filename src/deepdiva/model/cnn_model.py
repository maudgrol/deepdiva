#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import layers


class ConvModel(tf.keras.Model):

    def __init__(self, shape, num_outputs, **kwargs):
        super(ConvModel, self).__init__()
        self.shape = shape

        # Define all layers
        # Layer of convolutional block 1
        self.conv1 = tf.keras.layers.Conv2D(filters=32 ,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="same",
                                            activation="relu",
                                            name="Conv_1")
        self.max1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                 name="MaxPool_1")

        # Layer of convolutional block 2
        self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="same",
                                            activation="relu",
                                            name="Conv_2")
        self.max2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                 name="MaxPool_2")

        # Layer of convolutional block 3
        self.conv3 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="same",
                                            activation="relu",
                                            name="Conv_3")
        self.max3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                 name="MaxPool_3")

        # Layer of convolutional block 4
        self.conv4 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="same",
                                            activation="relu",
                                            name="Conv_4")
        self.max4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                 name="MaxPool_4")

        # Fully connected layers and dropout
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2,
                                               name="Dropout_1")
        self.flatten = tf.keras.layers.Flatten(name="Flatten_1")
        self.fc1 = tf.keras.layers.Dense(units=256,
                                         activation="relu",
                                         name="Dense_1")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.4,
                                               name="Dropout_2")
        self.fc2 = tf.keras.layers.Dense(units=num_outputs,
                                         activation="linear",
                                         name="Output_layer")

        self._build_graph()


    def _build_graph(self):
        self.build((None,) + self.shape)
        inputs = tf.keras.Input(shape=(self.shape))
        self.call(inputs)


    def call(self, input_tensor, training=None):
        # forward pass: convolutional block 1
        x = self.conv1(input_tensor)
        x = self.max1(x)

        # forward pass: convolutional block 2
        x = self.conv2(x)
        x = self.max2(x)

        # forward pass: convolutional block 3
        x = self.conv3(x)
        x = self.max3(x)

        # forward pass: convolutional block 4
        x = self.conv4(x)
        x = self.max4(x)

        # forward pass: dense layers, dropout and output
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x, training=training)
        return self.fc2(x)
