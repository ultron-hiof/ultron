# ultron
 Machine learning framework @hiof
 

## Use case
Framework for making machine learning easier. This framework utilizes:
* keras
* matplotlib
* OpenCV
* tqdm
* PlaidML (MacOS)
* TensorFlow (Windows)

## Installation of needed packages MacOS

Use the package manager [pip](https://pip.pypa.io/en/stable/) for installing needed packages.

```bash
pip install keras
pip install plaidml-keras plaidbench
pip install -U matplotlib
pip install opencv-python
pip install tqdm
```

### Setup GPU training with PlaidML on MacOS
Choose which accelerator you'd like to use (many computers, especially laptops, have multiple)
In the terminal of your python project (venv) write:

Step 1:
```bash
plaidml-setup
```

Now try benchmarking MobileNet:
```bash
plaidbench keras mobilenet
```



## Installation of needed packages Windows
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install needed packages.

```bash
pip install keras
pip install tensorflow
pip install -U matplotlib
pip install opencv-python
pip install tqdm
```

## Examples
load_x_dataset and load_y_dataset returns the specified dataset to X and the labels to y, ready to be used in your
project
```python
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

```

def label_img_dataset() is made so that you can easy convert a colored image dataset grayscale and resize it.
Be sure to name the categories the same as the folders where the images is located.
```python
from ultron.label.dataset import label_img_dataset


# all images of left is located in the folder user/dataset/left and so on.
categories = ['left', 'right', 'forward', 'backward', 'stop']

# This function call will create features.pickle and labels.pickle
label_img_dataset(datadir='user/dataset/', categories=categories, image_size=64, x_name='features', y_name='labels')

```

plot_model() will plot the training and validation accuarcy and loss depending on what the user specifies.
The function takes in the history object that is createt by the fit function provided by Keras for training a
model. These function calls will create two images acc.png and loss.png in the given location.
```python
from ultron.plot.graph import plot_model

# placeholder (fit in the function call of training a model in keras)
history = model.fit()

plot_model(history=history, metric='acc', name=NAME, save_location='models/model_name/acc.png')
plot_model(history=history, metric='loss', name=NAME, save_location='models/model_name/loss.png')

```
![acc of model](/resources_git/acc.png)

This is an example of the graphes provided by the plot_model() function call.
## Authors
- William Svea-Lochert
- Fredrik Lauritzen

 

