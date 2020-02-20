# ultron
 Machine learning framework @hiof
 

## Use case
Framework for making machine learning easier. This framework utilizes:
* keras
* matplotlib
* OpenCV
* PlaidML (MacOS)
* TensorFlow (Windows)

## Installation of needed packages MacOS

Use the package manager [pip](https://pip.pypa.io/en/stable/) for installing needed packages.

```bash
pip install keras
pip install plaidml-keras plaidbench
pip install -U matplotlib
pip install opencv-python
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
```


## Authors
- William Svea-Lochert
- Fredrik Lauritzen

 

