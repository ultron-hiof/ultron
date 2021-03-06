# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# plot images from given dataset
# ------------------------------------------------------- #
import matplotlib.pyplot as plt
from dataset.load.img.dataset import load_x_dataset


# Show an image from your dataset if the image 1x....
def show_img(index, filepath, img_size):
    # Loading dataset
    x = load_x_dataset(filepath)

    # Show image
    plt.imshow(x[index].reshape(img_size, img_size), cmap='gray')
    plt.show()