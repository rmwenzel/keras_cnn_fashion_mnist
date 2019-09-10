# keras_cnn_fashion_mnist
Shallow CNN for Fashion MNIST using Keras and SageMaker

## Overview

This project contains code and notebooks for training a custom CNN on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset either locally or in the cloud using Amazon SageMaker.

The CNN is written with Keras and Tensorflow backend.

## Training scripts

To train the network on your local machine:

```
python keras_cnn_fashion_mnist_local.py
```

To train using Amazon SageMaker use `keras_cnn_fashion_mnist_sm.py` and [script mode](https://github.com/aws-samples/amazon-sagemaker-script-mode)
This can be done locally or within a SageMaker notebook instance (see [notebooks](#Notebooks) for more details). 

The easiest way to get running a notebook instance is probably to fork this repo and link to the notebook instance.

## Notebooks

There are two Jupyter notebooks:

`keras_cnn_fashion_mnist_local.ipynb`: Explore the fashion MNIST dataset, train the CNN using local resources, and train using Sagemaker resources from local machine.

`keras_cnn_fashion_mnist_local.ipynb`: Train the CNN directly in a Sagemaker notebook instance.


## Environments.

There were a few challenges in setting up the necessary packages, so if you're running the scripts or notebooks locally, it's recommended to create a virtual environment directly from the included environment files

Using [`virtualenv`](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

```
python3 -m venv fashion
source env/bin/activate
pip install -r requirements.txt
```

Using [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

```
conda env create -f environment.yml
```

If you're running anything in a SageMaker notebook instance, you can use the built in `conda_python3` kernel, provided you install 