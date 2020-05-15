# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 20.02.2020
# Convert images to grayscale, resize them, label and
# exports the dataset as .pickle files for later use.
# ------------------------------------------------------- #
import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

# load in the data, convert to grey scale and resize the images.
# noinspection SpellCheckingInspection


def label_img_dataset(datadir, categories, img_size, x_name, y_name, rgb=None):
    training_data = []
    color_mode = None

    if rgb or rgb is None:
        color_mode = cv2.COLOR_BGR2RGB
    else:
        color_mode = cv2.IMREAD_GRAYSCALE

    for cat in categories:
        path = os.path.join(datadir, cat)
        class_num = categories.index(cat)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), color_mode)
                resize_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([resize_array, class_num])
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path, img))
            except Exception as e:
                print("general exception", e, os.path.join(path, img))

    append_data(training_data, img_size, x_name, y_name, rgb)


# Shuffle and append labels and features
# then calls export_dataset()
def append_data(data, img_size, x_name, y_name, rgb):
    color_channels = None
    random.shuffle(data)

    x = []
    y = []

    for features, label in data:
        x.append(features)
        y.append(label)
        features = None
        label = None

    data.clear()
    data = None # clear the data array to save system memory

    if rgb:
        color_channels = 3
    else:
        color_channels = 1

    x = np.array(x).reshape(-1, img_size, img_size, color_channels)

    export_dataset(x_name, x)
    export_dataset(y_name, y)


# export the dataset as .pickle for later use
def export_dataset(name, data):
    pickle_out = open(name + '.pickle', 'wb')
    pickle.dump(data, pickle_out, protocol=4)
    pickle_out.close()
