# keras_cnn_fashion_mnist
Shallow CNN for Fashion MNIST using Keras and SageMaker

## Overview

This project contains code and notebooks for training a custom CNN on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset either locally or in the cloud using Amazon SageMaker.

The CNN is written with Keras and Tensorflow backend. Generic and Fashion MNIST specific-version are implemented as classes in `cnn.py`.

## Directories

- `data` - Fashion MNIST data
- `models` - models saved during local training
- `sm_output` - Sagemaker job results 

## Training scripts

To train the network on your local machine:

```
python train_script_local.py
```

To train using Amazon SageMaker use `train_script_sagemaker.py` and [script mode](https://github.com/aws-samples/amazon-sagemaker-script-mode).
This can be done locally or within a SageMaker notebook instance (see [notebooks](#Notebooks) for more details)[^1]


## Notebooks

There is a Jupyter notebook `explore_data_and_model.ipynb` for exploration of the dataset and the model.
There are two Jupyter notebooks for training the model:

`train_model_local.ipynb`: Train the CNN using local resources, and train using Sagemaker resources from local machine.

`train_model_sagemaker.ipynb`: Train the CNN directly in a Sagemaker notebook instance.


## Environments.

If you're running the scripts or notebooks locally, it's recommended to create a virtual environment directly from the included environment files

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

If you're running anything in a SageMaker notebook instance, you can use the built-in `conda_python3` kernel, provided you install
`keras` [^2]


[^1]: The easiest way to get up and running in a SageMaker notebook instance is probably to fork this repo and link it to the notebook instance.
[^2]: At the time of writing, this wouldn't work using a Lifecycle configuration due to a timeout, but you can install from the notebooks.