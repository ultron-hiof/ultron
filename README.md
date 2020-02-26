# Ultron
 Machine learning framework based on keras with the help of other tools. 
 
The framework the group have in mind is made to make working with machine learning easier. 
Where the framework we make will make the users save time. Where the library with the functions 
is premade, and the users do not have to make them beforehand.

## Aim of the project

Usages of smart variable names
Short down work times
Low Barrier to Entry where users can choose the functionality they want.
Make it easier to work with dataset
Resize and greyscale
Picture sequence
Label dataset
Creating networks
Searching for networks 
Use network in a basic way

Background 
The group consists of Fredrik Lauritzen a 2nd year informatics student and William Svea-Lochert a 3rd year informatics student. Both have an interest in machine learning and wants to know more about it.

The background for this project is because William is currently working with his Bachelor thesis, where he is actively working machine learning and creating datasets. We wanted to make the process of working with machine learning easier, by having a framework that is a collection of some basic and some more advanced functionality.


## Other frameworks Ultron is built on top of
Framework for making machine learning easier. This framework utilizes:
* matplotlib
* OpenCV
* tqdm
* PlaidML

## Installation of needed packages

Use the package manager [pip](https://pip.pypa.io/en/stable/) for installing needed packages.

```bash
pip install keras
pip install plaidml-keras plaidbench
pip install -U matplotlib
pip install opencv-python
pip install tqdm
pip install termcolor
```

### Setup GPU training with PlaidML
Choose which accelerator you'd like to use (many computers, especially laptops, have multiple)
In the terminal of your python project (venv) write:

```bash
plaidml-setup
```

Now try benchmarking MobileNet:
```bash
plaidbench keras mobilenet
```


## Use cases

## Video to images
**video_to_images()** function call with take a folder structure that contais videos and convert the videos into images
in their new respective folder. The function call takes a input path for where on your machine the videos are located, 
an output path to where it will create a new folder three, an array containing all the folder names(categories) and the
image size you want for the images.

Example:

```python
from ultron.dataset.video_convertion.video import video_to_images

video_to_images(input_path='/Users/dataset/location_of_multiple_folders',
                output_path='/Users/dataset/output_folder',
                folders=['Dog', 'Cat'], img_size=200)
```

## Creating and using your Dataset

### Label dataset
**label_img_dataset()** is made so that you can easy convert a colored image dataset grayscale and resize it.
Be sure to name the categories the same as the folders where the images is located.

Example:
```python
from ultron.dataset.label.dataset import label_img_dataset


# all images of left is located in the folder user/dataset/left and so on.
categories = ['left', 'right', 'forward', 'backward', 'stop']

# This function call will create features.pickle and labels.pickle
label_img_dataset(datadir='user/dataset/', categories=categories, image_size=64, x_name='features', y_name='labels')

```




### Load dataset
**load_x_dataset()** and **load_y_dataset()** returns the specified dataset to X and the labels to y, ready to be used in your
project

Example:
```python
from ultron.dataset.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

```



## Plotting images

### Plot training and validation accuarcy & loss
**plot_model()** will plot the training and validation accuarcy and loss depending on what the user specifies.
The function takes in the history object that is createt by the fit function provided by Keras for training a
model. These function calls will create two images acc.png and loss.png in the given location. Image is 
shown below code snippet.

Example:
```python
from ultron.plot.graph import plot_model

# placeholder (fit in the function call of training a model in keras)
history = model.fit()

plot_model(history=history, metric='acc', name=NAME, save_location='models/model_name/acc.png')
plot_model(history=history, metric='loss', name=NAME, save_location='models/model_name/loss.png')

```
![acc of model](/resources_git/acc.png)


### Plot image from your dataset
**show_linear_img()** is a function call for the users of a fully connected neural network users. 
This function will show an image from your dataset. This is when your images has been converted to a
linear image this means that the image is on one axis (1x…)

Example:
```python
from ultron.plot.img import show_linear_img

# Plot the image in the given index and show it to the user
show_linear_img(index=43, filepath='user/project/dataset.pickle', img_size=64)

```
**show_img()** is a function call for the users of a Conv2d model, which uses normal images. The function
call will plot an image from your dataset with the given index.

Example:
```python
from ultron.plot.img import show_img

# Plot the image in the given index from the dataset set by the user, and show it to the user
show_img(index=43, filepath='user/project/dataset.pickle')

```
![acc of model](/resources_git/doggo.png)

# Models

## Create Conv2D model

**create_model()** is a function for create a Conv2D model, this function can take in multiple parameters or just one to
create a model that the user can train for their use.

Example where user specify only the input shape parameter:
```python
from ultron.model.conv import create_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# Creates a 'default' model with the input shape given by the user.
# Default output is 1 category.
model = create_model(shape=X.shape[1:])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

Example where user specify all parameters:
```python
from ultron.model.conv import create_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# Creates the model with all parameters given by the user
model = create_model(shape=X.shape[1:], conv_layers=2, conv_layer_size=32, activation_layer='relu', dense_layers=1, dense_layer_size=512, output_classes=2, output_activation='softmax')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

## Create a fully connected feed forward model

**create_model()** is a function for create a fully connected feed forward model, this function can take in multiple parameters or just one to
create a model that the user can train for their use.

Example where user specify only the input shape parameter:
```python
from ultron.model.fully_connected import create_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# Creates a 'default' model with the input shape given by the user.
# Default output is 1 category.
model = create_model(shape=X.shape[1:])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

Example where user specify all parameters:
```python
from ultron.model.fully_connected import create_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# Creates the model with all parameters given by the user
model = create_model(shape=X.shape[1:], dense_layers=2, dense_layer_size=512, activation_layer='relu', output_classes=2, output_activation='softmax')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```
## Network search
**network_search()** The usages for this function is to find the model that fits your project the best by training 
multiple models and using the plot functionality of this framework to show you the results.

Example to network search:
```python

```

## Authors
- William Svea-Lochert
- Fredrik Lauritzen

 

