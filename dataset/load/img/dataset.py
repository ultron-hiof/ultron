# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 20.02.2020
# Load greyscale img datasets into project
# ------------------------------------------------------- #
import pickle
import numpy as np
from termcolor import colored


def load_x_dataset(filepath):
    pickle_in = open(filepath, 'rb')
    x = pickle.load(pickle_in)
    x = x/255.0
    text = colored('X: features loaded', 'green')
    print(text)
    return x


def load_y_dataset(filepath):
    pickle_in = open(filepath, 'rb')
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    text = colored('y: labels loaded', 'green')
    print(text)
    return y
