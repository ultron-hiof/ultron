# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Sets PlaidML as backend with Keras on top.
# Proceeds then to load a given model, specify early stop
# callback for escaping overfittment. Then proceeds to train
# the given model for the specified amount of epochs, give
# a summary of the model then save the model and return the
# history object created by the fit() function call.
# ------------------------------------------------------- #

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model


# Train the given model
def train_model(X, y, epochs, batch_size, val_split, model, save_location):
    # Load the model from the given location
    model = load_model(model)  # Load model

    # Train model with early stop callback
    history = model.fit(X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=val_split)  # train the model

    # Prints model summary & save model to given location
    # model.summary()
    model.save(save_location + '.model')

    # Return history object
    return history
