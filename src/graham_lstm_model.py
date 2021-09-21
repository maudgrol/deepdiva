#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import layers

from graham_highway_layer import HighwayLayer

class LstmHighwayModel(tf.keras.Model):

    def __init__(self, shape, num_outputs, lstm_size=128, number_of_highway_layers=6, **kwargs):
        super(LstmHighwayModel, self).__init__()
        self.shape = shape

        # Define all layers in init
        self.lstm1 = layers.Bidirectional(
            layers.LSTM(lstm_size),
            input_shape=self.shape,
            merge_mode='concat'
        )

        self.dropout1 = layers.Dropout(0.2)

        self.dense1 = layers.Dense(
            64,
            activation='elu',
            activity_regularizer=tf.keras.regularizers.l2()
        )

        # Define highway layers
        self.highway_layers = []
        for i in range(number_of_highway_layers):
            highway_layer = HighwayLayer(
                activation='elu',
                transform_dropout=0.2,
                activity_regularizer=tf.keras.regularizers.l2()
            )
            self.highway_layers.append(highway_layer)

        self.dense2 = layers.Dense(
            num_outputs,
            activation='elu',
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_normal_initializer(stddev=0.01),
        )

        self._build_graph()

    def _build_graph(self):
        self.build((None,) + self.shape)
        inputs = tf.keras.Input(shape=self.shape)
        self.call(inputs)


    def call(self, input_tensor, training=False):
        # lstm layer
        x = self.lstm1(input_tensor)

        # dropout when training
        if training:
            x = self.dropout1(x)

        # dense layer 1
        x = self.dense1(x)
        
        # highway layers
        for highway_layer in self.highway_layers:
            x = highway_layer(x)

        return self.dense2(x)


