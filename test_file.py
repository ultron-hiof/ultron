import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataset.label.dataset import label_img_dataset
from plot.img import show_img

CATEGORIES = ['Dog', 'Cat']

label_img_dataset(datadir='C:/Users/lauri/Documents/Skole/Rammeverk/dataset', categories=CATEGORIES, img_size=100,
                  x_name='features', y_name='labels', rgb=False)


# C:\Users\lauri\Documents\Skole\Rammeverk\dataset