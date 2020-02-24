import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


def create_model(conv_layers=None, conv_layer_size=None, activation_layer=None, shape=None,
                 dense_layers=None, dense_layer_size=None,
                 output_classes=None, output_activation=None):
    if all(x is None for x in
           [conv_layers, conv_layer_size, activation_layer, dense_layer_size, dense_layers, output_activation,
            output_classes]) and shape is not None:
        model = create_model(conv_layers=2, conv_layer_size=32, activation_layer='relu', shape=shape, dense_layers=1,
                             dense_layer_size=512, output_classes=1, output_activation='softmax')

        return model
    elif shape is None:
        print('You must specify a input shape, use: shape=')
        return None
    else:
        # Model definition.
        model = Sequential()

        # Input layer
        model.add(Conv2D(conv_layer_size, kernel_size=(3, 3), activation=activation_layer, input_shape=shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Create Conv2D Layers.
        for x in range(conv_layers):
            model.add(Conv2D(conv_layer_size, kernel_size=(3, 3), activation=activation_layer))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        # Going from Conv2D to Fully connected.
        model.add(Flatten())

        # Creating Dense layers.
        for y in range(dense_layers):
            model.add(Dense(dense_layer_size, activation=activation_layer))

        # Output player
        model.add(Dense(output_classes, activation=output_activation))

        return model
