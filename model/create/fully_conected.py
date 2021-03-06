# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Sets PlaidML as backend with Keras on top.
# create a FF model
# ------------------------------------------------------- #
import os
from termcolor import colored

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Flatten


def create_model(dense_layers=None, dense_layer_size=None, activation_layer=None, shape=None,
                 output_classes=None, output_activation=None):
    if all(x is None for x in
           [dense_layers, dense_layer_size, activation_layer,
            output_classes, output_activation]) and shape is not None:

        model = create_model(dense_layers=2, dense_layer_size=512, activation_layer='relu', shape=shape,
                 output_classes=2, output_activation='softmax')

        return model

    elif shape is None:
        error = colored('INPUT ERROR: You must specify a input shape, use: shape=', 'red')
        print(error)
    else:
        model = Sequential()
        model.add(Flatten()) # turns dataset to 1 x total_pixels

        # input layer
        model.add(Dense(dense_layer_size, activation=activation_layer, input_shape=shape))
        # create dense layers
        for dense in range(dense_layers):
            model.add(Dense(dense_layer_size, activation=activation_layer))

        # output layer
        model.add(Dense(output_classes, activation=output_activation))
        print(colored('Model created!', 'green'))
        return model
