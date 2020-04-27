# Ultron
 Ultron is a framework that is made to make machine learning easier to work with. Ultron is based on several other 
 frameworks like Keras for the core machine learning tools, PlaidML to be able to train your network on any kind
 of hardware, OpenCv for image processing, matplotlib for plotting your training validation and accuracy, Numpy and 
 pickle of loading and exporting new datasets.
 
 Ultron is a collection of tools that you as the user normally would have to write yourself, but with ultron 
 the tools you need is at you fingertips. From creating you first machine learning model to retraining your models, 
 confusion matrix's, image processing, creating image dataset's and so much more!
 
The framework the group have in mind is made to make working with machine learning easier. 
Where the framework we make will make the users save time. Where the library with the functions 
is premade, and the users do not have to make them beforehand.

Ultron is very much aimed at developers who has not work that much or are brand new to the field. But Ultron contains some
nice to have tools that is good even for experienced developers who wants to shorten down work times and write fewer lines of code.


## Aim of the project

* Short down work times.
* Low Barrier to Entry where users can choose the functionality they want.
* Make it easier to create and work with dataset.
* Make image pre-processing easier.
* Create picture sequences from videos and images for datasets.
* easily create your own image dataset.
* Creating basic machine learning models.
* Neural network searching, find what network architecture fits best for your project.
* Retraining your models with a simple function call.
* Test your model easy with a simple function call, that gives you the information needed to make your model predictions better.


### Background 
The group consists of:
* Fredrik Lauritzen a 2nd year informatics student @Hiof.
* William Svea-Lochert a 3rd year informatics @ Hiof

We have an interest in machine learning and want to learn more about it, how it works and how to make it easier for someone that
has never tried developing it before. 

The background for this project is because William is currently working with his Bachelor thesis, where he is 
actively working machine learning and creating datasets. We wanted to make the process of working with 
machine learning easier, by having a framework that is a collection of some basic and some more advanced functionality.


## Other frameworks Ultron is built on top of
Ultron utilizes these frameworks:
* matplotlib
* OpenCV
* tqdm
* PlaidML
* Keras
* pickle
* sty
* termcolor

Ultron uses the flexsible platforms of Keras and PlaidML. By using PlaidML you are able to train on basically any GPU, 
and are you are not restricted to only use PlaidML. The models produces are plug-and-play with other machine learning 
frameworks like TensorFlow.

## Installation of needed packages

Use the package manager [pip](https://pip.pypa.io/en/stable/) for installing needed packages.

```bash
pip install plaidml-keras plaidbench
pip install -U matplotlib
pip install opencv-python
pip install tqdm
pip install termcolor
pip install sty
pip install pickle
```

### Setup GPU training with PlaidML
```diff
+ This is recomended if you have a dedicated GPU!
```
Choose which accelerator you'd like to use (many computers, especially laptops, have multiple)
In the terminal of your python project (venv) write:

```bash
plaidml-setup
```

* Enable experimental mode
* Select your graphics card

Now try benchmarking MobileNet:
```bash
plaidbench keras mobilenet
```
* You are now good to go :smile: 

# Use cases

## Video to images
**video_to_images()** function call with take a folder structure that contains videos and convert the videos into images
in their new respective folder. The function call takes a input path for where on your machine the videos are located, 
an output path to where it will create a new folder tree, an array containing all the folder names(categories) and the
image size you want for the images.

#### Parameters
* **`input_path`:** is the directory where your video files are located.
* **`output_path`:** is where the new images is to be saved.
* **`folders`:** are the folders which the images is split between so you can easily label them later. (The folders must be
created in advance by the user for now, update will come later where the function call will create the folders for you.)
* **`img_size`:** is the desired size of your image (For now the function creates a square image, update comming!)

#### Example code:

```python
from ultron.dataset.video_convertion.video import video_to_images

video_to_images(input_path='/Users/dataset/location_of_multiple_folders',
                output_path='/Users/dataset/output_folder',
                folders=['Dog', 'Cat'], img_size=200)
```


## Creating and loading your own Dataset

### Label dataset
**label_img_dataset()** is made so that you can easy label a image dataset, either colored or in Grayscale.
With just a simple function call, by the given location of the images, all your categories, image size, name of the new
.pickle files that the function generates and if you want the image to be in RGB or in Grayscale, you will have a dataset
ready to use!

#### Parameters
* **`datadir`:** is the directory where your image files are located.
* **`categories`:** is what you have named your features and the corresponding folders
* **`img_size`:** is the size you want you square image to be.
* **`x_name`:** is the name of the features pickle file that the function produces.
* **`y_name`:** is the name of the labels pickle file that the function produces.
* **`rgb`:** is if you want your dataset to be RGB or Grayscale. Default when left out is RGB. To specify Grayscale set rgb to False.

#### Example code:
```python
from ultron.dataset.label.dataset import label_img_dataset


# all images of left is located in the folder user/dataset/left and so on.
categories = ['left', 'right', 'forward', 'backward', 'stop']

# This function call will create features.pickle and labels.pickle
label_img_dataset(datadir='user/dataset/', categories=categories, img_size=64, 
                  x_name='features', y_name='labels', rgb=False)

```




### Load dataset
**load_x_dataset()** and **load_y_dataset()** returns the specified dataset to X and the labels to y, ready to be used in your
project. This function takes in a .pickle file as its argument, and works perfectly with the **label_img_dataset()** function.
By running this function you will have loaded you dataset with its features and labels ready to use!

**load_x_dataset()** is for loading your images/features.
#### Parameters
* **`filepath`:** is the location of your features .pickle file.

**load_y_dataset()** is for loading your labels for your images/features.
#### Parameters
* **`filepath`:** is the location of your labels .pickle file.

#### Example code:
```python
from ultron.dataset.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

```



## Plotting images

### Plot training and validation accuracy & loss
**plot_model()** will plot the training and validation accuracy and loss depending on what the user specifies.
The function takes in the history object that is created by the fit function provided by Keras for training a
model. These function calls will create two images acc.png and loss.png in the given location. Image is 
shown below code snippet.

#### Parameters
* **`history`:** is the history object produced by Keras own fit function.
* **`metric`:** is the metric you want to plot, either accuracy or loss.
* **`name`:** is the title of for the plot.
* **`save_location`:** is where you want to save the images of the plot.


#### Example code:
```python
from ultron.plot.graph import plot_model

# placeholder (fit in the function call of training a model with keras)
history = model.fit()

plot_model(history=history, metric='acc', name='model_acc', save_location='models/acc.png')
plot_model(history=history, metric='loss', name='model_loss', save_location='models/loss.png')

```
![acc of model](/resources_git/acc.png)


### Plot image from your dataset
**show_img()** This function will show an image from your dataset. This is useful for you to see what info you are
giving your model to train on. This function is mainly used to verify if your data is correct.

#### Parameters
* **`index`:** is the index of what image you want to plot in your dataset.
* **`filepath`:** is the path to where your features .pickle file is located.
* **`img_size`:** is the size of the image you want to plot.

#### Example code:
```python
from ultron.plot.img import show_img

# Plot the image in the given index and show it to the user
show_img(index=43, filepath='user/project/dataset.pickle', img_size=64)

```
![acc of model](/resources_git/doggo.png)

# Models

## Create Conv2D model

**create_model()** is a function for create a Conv2D model, this function can take in multiple parameters or just one to
create a model that the user can train for their use. This is a general model creation tool, for basic models so you as
the user can easier learn how to make a model. 

If you choose to fill in more parameters you can make a slightly more advanced model, but at the moment still a fairly 
basic model.

#### Parameters
* **`shape`:** is the shape of the data that the model is receiving, if you choose to use Ultron's **load_x_dataset()**
function call, it will in most cases work with only specifying the shape as `X.shape[1:]`
* **`conv_layers`:** how many Conv2D layers that is going to be in the model.
* **`conv_layer_size`:** how big the Conv2D layers should be.
* **`activation_layer`:** activation layers for the model.
* **`dense_layers`:** how many Dense layers there is going to be in the model.
* **`dense_layer_size`:** size of the Dense layers.
* **`output_classes`:** how many labels does your dataset have.
* **`output_activation`:** activation layer for the output layer in the neural network.

#### Example where user specify only the input shape parameter:
```python
from ultron.model.create.conv import create_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# Creates a 'default' model with the input shape given by the user.
# Default output is 1 category.
model = create_model(shape=X.shape[1:])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=X, y=y, epochs=10, validation_split=0.3)
model.save('model_name.model')

model.summary()

```

#### Example where user specify all parameters:
```python
from ultron.model.create.conv import create_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# Creates the model with all parameters given by the user
model = create_model(shape=X.shape[1:], conv_layers=2, conv_layer_size=32, 
                     activation_layer='relu', dense_layers=1, dense_layer_size=512, 
                     output_classes=2, output_activation='softmax')
                     
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=X, y=y, epochs=10, validation_split=0.3)
model.save('model_name.model')

model.summary()

```

## Create a fully connected model

**create_model()** is a function for create a fully connected feed forward model, this function can take in multiple parameters or just one to
create a model that the user can train for their use.

```diff
- To use a fully connected model your dataset must be reshaped to 1 axsis!
```

#### Parameters
* **`shape`:** is the shape of the data that the model is receiving, if you choose to use Ultron's **load_x_dataset()**
function call, it will in most cases work with only specifying the shape as `X.shape[1:]`
* **`dense_layers`:** how many Dense layers there is going to be in the model.
* **`dense_layer_size`:** size of the Dense layers.
* **`activation_layer`:** activation layers for the model.
* **`output_classes`:** how many labels does your dataset have.
* **`output_activation`:** activation layer for the output layer in the neural network.

#### Example where user specify only the input shape parameter:
```python
from ultron.model.create.fully_connected import create_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# Creates a 'default' model with the input shape given by the user.
# Default output is 1 category.
model = create_model(shape=X.shape[1:])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=X, y=y, epochs=10, validation_split=0.3)
model.save('model_name.model')

model.summary()
```

Example where user specify all parameters:
```python
from ultron.model.create.fully_connected import create_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# Creates the model with all parameters given by the user
model = create_model(shape=X.shape[1:], dense_layers=2, dense_layer_size=512, 
                     activation_layer='relu', output_classes=2, output_activation='softmax')

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=X, y=y, epochs=10, validation_split=0.3)
model.save('model_name.model')

model.summary()
```
## Network search
### Conv2d model search
**model_search()** The usages for this function is to find the model that fits your project the best by training 
multiple models and using the plot functionality of this framework to show you the results. The graphs will be saved as multiple
files in your project directory so that you can compare the results.

#### Parameters
* **`shape`:** is the shape of the data that the model is receiving, if you choose to use Ultron's **load_x_dataset()**
function call, it will in most cases work with only specifying the shape as `X.shape[1:]`
* **`conv_layers`:** array with different amount of Conv2D layers that is going to be in the models.
* **`conv_layer_size`:** array with different sizes for how big the Conv2D layers should be.
* **`activation_layer`:** activation layers for the model.
* **`dense_layers`:** array with different amount of how many Dense layers there is going to be in the models.
* **`output_classes`:** how many labels does your dataset have.
* **`output_activation`:** activation layer for the output layer in the neural network.
* **`X`:** your features .pickle file.
* **`y`:** your labels .pickle file.


```diff
- You will need to have a dataset ready to be able to use the model search!
```

#### Example with only dataset and shape:
By calling **model_search()** with only the shape and the dataset, the function call will stick to default values.
```python
from ultron.model.search.conv2d_search import model_search
from ultron.load.img.dataset import load_x_dataset, load_y_dataset


X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

model_search(shape=X.shape[1:], X=X, y=y)
```
#### Example of Conv2D network search with all parameters:
```python
from ultron.model.search.conv2d_search import model_search
from ultron.load.img.dataset import load_x_dataset, load_y_dataset


X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

model_search(conv_layers=[1, 2, 3], layer_sizes=[32, 64, 128], activation_layer='relu', shape=X.shape[1:],
                     dense_layers=[0, 1, 2], output_classes=2, output_activation='softmax', X=X, y=y)

```

### Fully Connected model search

#### Parameters
* **`shape`:** is the shape of the data that the model is receiving, if you choose to use Ultron's **load_x_dataset()** function call, it will in most cases work with only specifying the shape as `X.shape[1:]`
* **`dense_layers`:** array with different amount of how many Dense layers there is going to be in the models.
* **`activation_layer`:** activation layers for the model.
* **`output_classes`:** how many labels does your dataset have.
* **`output_activation`:** activation layer for the output layer in the neural network.
* **`X`:** your features .pickle file.
* **`y`:** your labels .pickle file.

#### Example with only dataset and shape:
By calling **model_search()** with only the shape and the dataset, the function call will stick to default values.

```python
from ultron.model.search.ff_search import model_search
from ultron.load.img.dataset import load_x_dataset, load_y_dataset


X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

model_search(shape=X.shape[1:], X=X, y=y)

```


#### Example of fully connected network search with all parameters:
```python
from ultron.model.search.ff_search import model_search
from ultron.load.img.dataset import load_x_dataset, load_y_dataset


X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

model_search(dense_layers=[1, 2, 3], layer_sizes=[32, 64, 128], 
             activation_layer='relu', shape=shape, output_classes=2, 
             output_activation='softmax', X=X, y=y)

```

## Retrain your model

**train_model()** function call will load your previously saved model and train it with the given dataset specified by 
the user for the amount of time specified. The function returns the history object given by the **fit()** function call
so that the user can plot the training & validation accuracy & loss.

#### Parameters
* **`X`:** your features .pickle file.
* **`y`:** your labels .pickle file.
* **`epoches`:** how many epoches the model should train for.
* **`val_split`:** how many percent of the dataset should be used for validation in each epoch of training. 
* **`model`:** the model you would like to train again.
* **`save_location`:** is where you would like to save the new model. (if you want to save it in the project dir
only specify a name for the new model.)


#### Example code:

```python
from ultron.model.training.retrain_model import train_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset
from ultron.plot.graph import plot_model


X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# This function call will train a model given by the user and return the 
# history obj from the fit() function call
history = train_model(X=X, y=y, epochs=100, val_split=0.3, model='user/location/model.model', 
                      save_location='user/location/new_location/new_model')

# plotting the training and validation loss & accuracy        
plot_model(history=history, metric='acc', name='NAME', save_location='models/acc.png')
plot_model(history=history, metric='loss', name='NAME', save_location='models/loss.png')

```

## Test your model with testdata

**predict()** runs two predictions on your model with the specified dataset given by 
the user. The function prints a Confusion matrix, the loss and accuracy of the model on the given dataset.

* **`X`:** your features .pickle file.
* **`y`:** your labels .pickle file.
* **`model_location`:** specify your .model file.
* **`categories`:** string array of your categories/labels. 

#### Example code:
```python
from ultron.model.training.retrain_model import train_model
from ultron.load.img.dataset import load_x_dataset, load_y_dataset
from ultron.plot.graph import plot_model
from ultron.model.test_model.validate_model import predict


X = load_x_dataset(filepath='user/project/file')
y = load_y_dataset(filepath='user/project/file')

# This function call will train a model given by the user and return the 
# history obj from the fit() function call
history = train_model(X=X, y=y, epochs=100, val_split=0.3, model='user/location/model.model', 
                      save_location='user/location/new_location/new_model')

# Datasets used for testing
test_x = load_x_dataset(filepath='user/project/file')
test_y = load_y_dataset(filepath='user/project/file')

# The categories on the dataset, in the correct order form when they where created.
categories = ['dog', 'cat']

# Validate your model on a testset, this function will print a Confusion matrix & 
# the loss and accuracy of the prediction done that was done
predict(model_location='user/location/new_location/new_model.model', 
               categories=categories, X=test_x, y=test_y)


```

## Authors
- William Svea-Lochert
- Fredrik Lauritzen :neckbeard: 

 

