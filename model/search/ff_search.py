# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Sets PlaidML as backend with Keras on top.
# function to search for the best architecture for your model
# ------------------------------------------------------- #
import os
from termcolor import colored
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from model.create.fully_conected import create_model
from plot.graph import plot_model


def model_search(dense_layers=None, layer_sizes=None, activation_layer=None, shape=None,
                 output_classes=None, output_activation=None, X=None, y=None):

    if all(x is None for x in
           [layer_sizes, activation_layer, dense_layers, output_activation,
            output_classes]) and shape is not None and X is not None and y is not None:

        model_search(dense_layers=[1, 2, 3], layer_sizes=[32, 64, 128], activation_layer='relu', shape=shape,
                     output_classes=2, output_activation='softmax', X=X, y=y)
    elif X is None:
        print(colored('Features X has not been specified', 'red'))
    elif y is None:
        print(colored('Labels y has not been specified', 'red'))
    elif X is None and y is None:
        print(colored('Features X and labels y has not been specified', 'red'))
    elif shape is None:
        print(colored('Input shape as not been specified', 'red'))
    else:
        print(colored('This may take a long time, depending on how many models has to be trained and tested', 'blue'))
        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
                print(colored('Creating ' + NAME, 'green'))

                model = create_model(dense_layers=dense_layer, activation_layer=activation_layer,
                                     shape=shape, dense_layer_size=layer_size,
                                     output_classes=output_classes, output_activation=output_activation)

                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                print(colored('Compiling model ' + NAME + ' using Loss: sparse_categorical_crossentropy and Optimizer: adam', 'yellow'))
                history = model.fit(X, y, batch_size=100, epochs=10, verbose=1, validation_split=0.3)

                # Plot training & validation accuracy & loss values
                plot_model(history, 'acc', NAME, 'model-' + NAME + 'acc.png')
                plot_model(history, 'loss', NAME, 'model-' + NAME + 'loss.png')
