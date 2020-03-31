import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataset.video_convertion.video import video_to_images
from dataset.label.dataset import label_img_dataset
from plot.img import show_img

# video_to_images(input_path='/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/unprocessed/dataset_4/',
#                 output_path='/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_4/',
#                 folders=['forward', 'backward', 'left', 'right', 'stop'], img_size=224)

label_img_dataset(datadir='/Users/william/Documents/gitHub/B20IT38/greenscreen_data/dataset_1_room',
                  categories=["forward", "right", "left", "backward", "stop", "still"], img_size=100,
                  x_name='X', y_name='y', rgb=False)

# show_linear_img(2, 'X.pickle', 100)
