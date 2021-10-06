#!/usr/bin/env python
import tensorflow as tf

# Custom loss function root mean squared error
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

