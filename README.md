# keras_cnn_fashion_mnist
Shallow CNN for Fashion MNIST using Keras and SageMaker

## Overview

This project contains code and notebooks for training a custom CNN on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset either locally or in the cloud using Amazon SageMaker.

The CNN is written with Keras and Tensorflow backend. Generic and Fashion MNIST specific-versions are implemented as classes in `cnn.py`.

## Directories

- `data/` - Fashion MNIST data files
- `models/keras_checkpoints` - Keras checkpoints 

## Training scripts

To train the network on your local machine:

```
python train_script_local.py
```

To train in the cloud using Amazon SageMaker use `train_script_sagemaker.py` and [script mode](https://github.com/aws-samples/amazon-sagemaker-script-mode).
This can be done locally or it can be done in a Sagemaker notebook instance [^1].

## Notebooks

There are two Jupyter notebooks:

- `explore_data_and_model.ipynb` -- An introductio to the dataset and default model
- `train_tune_test.ipynb` - training, tuning and testing the model using Sagemaker resources.

Note `.html` versions of these notebooks are included for viewing convenient since `.ipynb` files don't render nicely on GitHub.


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

[^2]: At the time of writing, this wouldn't work using a Lifecycle configuration due to a timeout, but you can install directly from a Python notebook within the instance using `! conda install --name conda_python3 keras`