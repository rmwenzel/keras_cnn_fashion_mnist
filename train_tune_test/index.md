---
layout: page
title: Train, tune and test the CNN
---

[Previously]({{site.baseurl}}/explore_data_and_model) we explored the Fashion MNIST image data set and the CNN model used to classify these images. 

Now we'll train the model, tune it a bit, and finally test it.

We're going to use Amazon's Sagemaker cloud service to overcome local resource limitations.
We'll take advantage of its convenient [Python SDK](https://sagemaker.readthedocs.io/en/stable/) which manages AWS resources for us behind the scenes. A [`ml.p3.2xlarge` instance](https://aws.amazon.com/sagemaker/pricing/instance-types/) will significantly speed up training and choosing managed spot instances will yield considerable savings (usually 60-70%).

## Contents

- [Setup](#setup)
- [Training](#training)
  - [Set up s3](#setyp-up-s3)
  - [Run a single training job](#run-a-single-training-job)
  - [Evaluate training job](#evaluate-training-job)
    - [Download Keras checkpoints and history from s3](#download-keras-checkpoints-and-history-from-s3)
    - [Analyze training history](#analyze-training-history)
- [Tuning](#tuning)
  - [Sagemaker automatic model tuning](#sagemaker-automatic-model-tuning)
  - [Analyze tuning job result](#analyze-tuning-job-results)
- [Testing](#testing)
- [Conclusions](#conclusions)


## Setup


```python
import numpy as np
import pandas as pd
import os
import sagemaker
import boto3
import h5py

%matplotlib inline
import matplotlib.pyplot as plt

from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from cnn import FashionMNISTCNN as fmc

# filter out FutureWarnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Supress Tensorflow Warnings
import tensorflow.compat.v1.logging as logging
logging.set_verbosity(logging.ERROR)
```

    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /anaconda3/envs/fashion/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    Using TensorFlow backend.


## Training

Sagemaker will run the training script inside a (prebuilt) Docker container and will pull data from an s3 bucket we specify. The container will be torn down on completion of the training job but we can send container files to an s3 bucket before that. In particular, we'll send the validation accuracy improvement checkpoints and training history generated by our training script `train_script_sagemaker.py`.

We'll use the same s3 bucket for all of this. First we'll upload local data to the bucket, then create a directory for storing keras checkpoints and history. Finally we'll specify a path for the "model artifacts" of the training job, i.e. anything saved in the `opt/ml/model` directory of the training job container. In our case, this is just the Tensorflow serving model.



### Set up s3


```python
# Session info
sess = sagemaker.Session()
role_name = '<YOUR IAM ROLE NAME>'
bucket_name = '<YOUR BUCKET NAME>'

# upload data to s3
training_input_path   = sess.upload_data('data/train.hdf5', bucket=bucket_name, key_prefix='data')
validation_input_path = sess.upload_data('data/val.hdf5', bucket=bucket_name, key_prefix='data')
test_input_path = sess.upload_data('data/test.hdf5', bucket=bucket_name, key_prefix='data')
```


```python
# create checkpoint directory in s3
try:
    with open('models/keras_checkpoints/dummy.txt', 'x') as f:
        f.write('This is a dummy file')
except OSError:
    pass

checks_output_path = sess.upload_data('models/keras_checkpoints/dummy.txt', bucket=bucket_name, key_prefix='keras-checkpoints')
checks_output_path = os.path.dirname(checks_output_path)

# s3 path for job output
job_output_path = 's3://{}/'.format(bucket_name)
```

### Run a single training job

We'll run a single Sagemaker training job using the [default model]('./explore_data_and_model.ipynb/#keras-cnn-model-for-classification')

We use a `sagemaker.tensorflow.Tensorflow` estimator for this training job. We'll track loss and accuracy metrics for both training and validation data, which keras tracks by default. 

Note that our output path for keras checkpoints gets passed in as a hyperparameter.


```python
# objective and metric
metric_definitions = [{'Name': 'acc',
                       'Regex': 'acc: ([0-9\\.]+)'},
                      {'Name': 'val_acc',
                       'Regex': 'val_acc: ([0-9\\.]+)'},
                      {'Name': 'loss',
                       'Regex': 'loss: ([0-9\\.]+)'},
                      {'Name': 'val_loss',
                       'Regex': 'val_loss: ([0-9\\.]+)'}]


hyperparameters = {'epochs': 100, 'batch-size': 100, 'drop-rate': 0.5,
                   'checks-out-path': checks_output_path}

# create sagemaker estimator
tf_estimator = TensorFlow(entry_point='train_script_sagemaker.py', 
                          role=role_name,
                          train_volume_size=5,
                          train_instance_count=1, 
                          train_instance_type='ml.p3.2xlarge',
                          train_use_spot_instances=True,
                          train_max_wait=86400,
                          output_path=job_output_path,
                          framework_version='1.14', 
                          py_version='py3',
                          script_mode=True,
                          hyperparameters=hyperparameters,
                          metric_definitions=metric_definitions
                         )

paths = {'train': training_input_path, 'val': validation_input_path,
         'test': test_input_path, 'checks': checks_output_path}
```


```python
# train estimator asynchronously
tf_estimator.fit(paths, wait=False)
```

### Evaluate training job

#### Download Keras checkpoints and history from s3

Now we pull the keras checkpoints and history down from s3.


```python
def download_checks_from_s3(checks_output_path):
    s3_resource = boto3.resource('s3')
    bucket_name = os.path.dirname(checks_output_path).split('//')[1]
    prefix = os.path.basename(checks_output_path)
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix = prefix):
        local_dir = 'models/keras_checkpoints'
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        local_file = os.path.join(local_dir, 
                                  os.path.basename(obj.key))
        bucket.download_file(obj.key, local_file)

# delete any preexisting checkpoints
! rm models/keras_checkpoints/*
download_checks_from_s3(checks_output_path)
```

#### Analyze training history 

We'll plot the keras training history


```python
history_df = pd.read_csv('models/keras_checkpoints/FashionMNISTCNN-history.csv')
history_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>val_loss</th>
      <th>val_acc</th>
      <th>loss</th>
      <th>acc</th>
      <th>lr</th>
      <th>epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.333004</td>
      <td>0.8778</td>
      <td>0.520144</td>
      <td>0.82050</td>
      <td>0.001</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.263404</td>
      <td>0.9033</td>
      <td>0.316812</td>
      <td>0.88600</td>
      <td>0.001</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.247057</td>
      <td>0.9091</td>
      <td>0.268965</td>
      <td>0.90370</td>
      <td>0.001</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.297089</td>
      <td>0.8980</td>
      <td>0.240364</td>
      <td>0.91154</td>
      <td>0.001</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.251827</td>
      <td>0.9074</td>
      <td>0.221876</td>
      <td>0.92054</td>
      <td>0.001</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_history(history_df):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plt.plot('epoch', 'loss', data=history_df, label='train_loss')
    plt.plot('epoch', 'val_loss', data=history_df, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot('epoch', 'acc', data=history_df, label='train_acc')
    plt.plot('epoch', 'val_acc', data=history_df, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
plot_history(history_df)
```


![png](train_tune_test_files/train_tune_test_19_0.png)



```python
acc_max = history_df.loc[history_df['acc'].idxmax(), :]
print('Maximum training accuracy epoch: \n{}'.format(acc_max))
```

    Maximum training accuracy epoch: 
    val_loss     0.275882
    val_acc      0.935300
    loss         0.040384
    acc          0.985260
    lr           0.001000
    epoch       42.000000
    Name: 41, dtype: float64



```python
val_acc_max = history_df.loc[history_df['val_acc'].idxmax(), :]
print('Maximum validation accuracy epoch: \n{}'.format(val_acc_max))
```

    Maximum validation accuracy epoch: 
    val_loss     0.237604
    val_acc      0.938600
    loss         0.055796
    acc          0.979620
    lr           0.001000
    epoch       32.000000
    Name: 31, dtype: float64



```python
# Validation accuracy epochs in descending order
history_df.drop(columns=['val_loss', 'loss']).sort_values(by='val_acc', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>val_acc</th>
      <th>acc</th>
      <th>lr</th>
      <th>epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>31</td>
      <td>0.9386</td>
      <td>0.97962</td>
      <td>0.001</td>
      <td>32</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.9369</td>
      <td>0.98394</td>
      <td>0.001</td>
      <td>41</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.9367</td>
      <td>0.97210</td>
      <td>0.001</td>
      <td>24</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.9365</td>
      <td>0.98308</td>
      <td>0.001</td>
      <td>39</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.9357</td>
      <td>0.97814</td>
      <td>0.001</td>
      <td>30</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.9356</td>
      <td>0.98280</td>
      <td>0.001</td>
      <td>40</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0.9353</td>
      <td>0.98526</td>
      <td>0.001</td>
      <td>42</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.9352</td>
      <td>0.97368</td>
      <td>0.001</td>
      <td>25</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.9351</td>
      <td>0.98288</td>
      <td>0.001</td>
      <td>37</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.9350</td>
      <td>0.97824</td>
      <td>0.001</td>
      <td>31</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.9345</td>
      <td>0.97972</td>
      <td>0.001</td>
      <td>34</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.9342</td>
      <td>0.98230</td>
      <td>0.001</td>
      <td>35</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.9340</td>
      <td>0.97988</td>
      <td>0.001</td>
      <td>33</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.9339</td>
      <td>0.96342</td>
      <td>0.001</td>
      <td>19</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.9339</td>
      <td>0.97800</td>
      <td>0.001</td>
      <td>29</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.9339</td>
      <td>0.97544</td>
      <td>0.001</td>
      <td>28</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.9335</td>
      <td>0.96752</td>
      <td>0.001</td>
      <td>21</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.9334</td>
      <td>0.97144</td>
      <td>0.001</td>
      <td>23</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.9317</td>
      <td>0.97508</td>
      <td>0.001</td>
      <td>27</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.9313</td>
      <td>0.98254</td>
      <td>0.001</td>
      <td>38</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.9310</td>
      <td>0.95914</td>
      <td>0.001</td>
      <td>16</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.9306</td>
      <td>0.95532</td>
      <td>0.001</td>
      <td>15</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.9306</td>
      <td>0.96310</td>
      <td>0.001</td>
      <td>18</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.9300</td>
      <td>0.96974</td>
      <td>0.001</td>
      <td>22</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.9297</td>
      <td>0.98172</td>
      <td>0.001</td>
      <td>36</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.9288</td>
      <td>0.96678</td>
      <td>0.001</td>
      <td>20</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.9281</td>
      <td>0.94712</td>
      <td>0.001</td>
      <td>12</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.9281</td>
      <td>0.93736</td>
      <td>0.001</td>
      <td>9</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.9279</td>
      <td>0.97398</td>
      <td>0.001</td>
      <td>26</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.9265</td>
      <td>0.96062</td>
      <td>0.001</td>
      <td>17</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.9261</td>
      <td>0.95488</td>
      <td>0.001</td>
      <td>14</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.9258</td>
      <td>0.94466</td>
      <td>0.001</td>
      <td>11</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.9216</td>
      <td>0.94084</td>
      <td>0.001</td>
      <td>10</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.9114</td>
      <td>0.94970</td>
      <td>0.001</td>
      <td>13</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.9091</td>
      <td>0.90370</td>
      <td>0.001</td>
      <td>3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.9074</td>
      <td>0.92054</td>
      <td>0.001</td>
      <td>5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.9033</td>
      <td>0.88600</td>
      <td>0.001</td>
      <td>2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.9031</td>
      <td>0.92906</td>
      <td>0.001</td>
      <td>7</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.9002</td>
      <td>0.93370</td>
      <td>0.001</td>
      <td>8</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.8980</td>
      <td>0.91154</td>
      <td>0.001</td>
      <td>4</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.8947</td>
      <td>0.92276</td>
      <td>0.001</td>
      <td>6</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.8778</td>
      <td>0.82050</td>
      <td>0.001</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We note that $93\%$ accuracy first occured roughly during epochs 15-18, and didn't improve much thereafter.

The last epoch where improvement occured was epoch 32, and since the default model has an early stopping patience of 10 epochs, we know it didn't improve from epochs 32-42 and training stopped after epoch 42.


```python
# Validation loss epochs in descending order
history_df.drop(columns=['val_acc', 'acc']).sort_values(by='val_loss', ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>val_loss</th>
      <th>loss</th>
      <th>lr</th>
      <th>epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>14</td>
      <td>0.200692</td>
      <td>0.122466</td>
      <td>0.001</td>
      <td>15</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.204977</td>
      <td>0.168892</td>
      <td>0.001</td>
      <td>9</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.206213</td>
      <td>0.151945</td>
      <td>0.001</td>
      <td>11</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.206824</td>
      <td>0.144087</td>
      <td>0.001</td>
      <td>12</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.207615</td>
      <td>0.112607</td>
      <td>0.001</td>
      <td>16</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.212421</td>
      <td>0.088360</td>
      <td>0.001</td>
      <td>21</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.214470</td>
      <td>0.099285</td>
      <td>0.001</td>
      <td>19</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.216433</td>
      <td>0.124686</td>
      <td>0.001</td>
      <td>14</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.220042</td>
      <td>0.107955</td>
      <td>0.001</td>
      <td>17</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.221858</td>
      <td>0.100688</td>
      <td>0.001</td>
      <td>18</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.223961</td>
      <td>0.078725</td>
      <td>0.001</td>
      <td>23</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.223996</td>
      <td>0.162515</td>
      <td>0.001</td>
      <td>10</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.224511</td>
      <td>0.073269</td>
      <td>0.001</td>
      <td>25</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.229332</td>
      <td>0.091966</td>
      <td>0.001</td>
      <td>20</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.231225</td>
      <td>0.082998</td>
      <td>0.001</td>
      <td>22</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.231908</td>
      <td>0.076089</td>
      <td>0.001</td>
      <td>24</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.236227</td>
      <td>0.065881</td>
      <td>0.001</td>
      <td>28</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.237604</td>
      <td>0.055796</td>
      <td>0.001</td>
      <td>32</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.241979</td>
      <td>0.068059</td>
      <td>0.001</td>
      <td>27</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.243946</td>
      <td>0.061642</td>
      <td>0.001</td>
      <td>30</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.244725</td>
      <td>0.055597</td>
      <td>0.001</td>
      <td>33</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.247057</td>
      <td>0.268965</td>
      <td>0.001</td>
      <td>3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.251827</td>
      <td>0.221876</td>
      <td>0.001</td>
      <td>5</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.252233</td>
      <td>0.061005</td>
      <td>0.001</td>
      <td>31</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.255708</td>
      <td>0.055532</td>
      <td>0.001</td>
      <td>34</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.255919</td>
      <td>0.135069</td>
      <td>0.001</td>
      <td>13</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.256263</td>
      <td>0.045249</td>
      <td>0.001</td>
      <td>41</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.256528</td>
      <td>0.070083</td>
      <td>0.001</td>
      <td>26</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.258491</td>
      <td>0.046385</td>
      <td>0.001</td>
      <td>40</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.258976</td>
      <td>0.047897</td>
      <td>0.001</td>
      <td>37</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.259441</td>
      <td>0.048968</td>
      <td>0.001</td>
      <td>35</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.261749</td>
      <td>0.047258</td>
      <td>0.001</td>
      <td>38</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.262236</td>
      <td>0.060982</td>
      <td>0.001</td>
      <td>29</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.263404</td>
      <td>0.316812</td>
      <td>0.001</td>
      <td>2</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.264150</td>
      <td>0.045217</td>
      <td>0.001</td>
      <td>39</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.272420</td>
      <td>0.194994</td>
      <td>0.001</td>
      <td>7</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0.275882</td>
      <td>0.040384</td>
      <td>0.001</td>
      <td>42</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.279350</td>
      <td>0.050247</td>
      <td>0.001</td>
      <td>36</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.285503</td>
      <td>0.182704</td>
      <td>0.001</td>
      <td>8</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.295849</td>
      <td>0.209872</td>
      <td>0.001</td>
      <td>6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.297089</td>
      <td>0.240364</td>
      <td>0.001</td>
      <td>4</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.333004</td>
      <td>0.520144</td>
      <td>0.001</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We also note that validation loss was also at an absolute minimum at epoch 14, so here is likely where the model begins to overfit.

## Tuning

### Sagemaker automatic model tuning

We'll use Sagemaker's built-in hyperparameter optimization to try to find a model 
with better validation accuracy. We'll use the (default) Bayesian strategy to search the hyperparameter space efficiently.


```python
# architecture hyperparameter spaces
conv0_hps = {'conv0_pad': IntegerParameter(1, 3),
             'conv0_channels': IntegerParameter(24, 32),
             'conv0_filter': IntegerParameter(2, 4),
             'conv0_stride': IntegerParameter(1, 3),
             'conv0_pool': IntegerParameter(1, 3),
            }
conv1_hps = {'conv1_pad': IntegerParameter(1, 3),
             'conv1_channels': IntegerParameter(48, 64),
             'conv1_filter': IntegerParameter(2, 4),
             'conv1_stride': IntegerParameter(1, 3),
             'conv1_pool': IntegerParameter(1, 3),
            }
conv2_hps = {'conv2_pad': IntegerParameter(1, 3),
             'conv2_channels': IntegerParameter(96, 128),
             'conv2_filter': IntegerParameter(2, 4),
             'conv2_stride': IntegerParameter(1, 3),
             'conv2_pool': IntegerParameter(1, 3),
            }
fc0_hps = {'fc0_neurons': IntegerParameter(200, 300)}
fc1_hps = {'fc1_neurons': IntegerParameter(200, 300)}

hyperparameter_ranges = {**conv0_hps, **conv1_hps, **conv2_hps, **fc0_hps, **fc1_hps}

# objective and metric
objective_metric_name = 'val_acc'
objective_type = 'Maximize'
metric_definitions = [{'Name': 'val_acc',
                       'Regex': 'best_val_acc: ([0-9\\.]+)'}]

# tuner
tuner = HyperparameterTuner(tf_estimator,
                            objective_metric_name,
                            hyperparameter_ranges,
                            metric_definitions,
                            max_jobs=10,
                            max_parallel_jobs=1,
                            objective_type=objective_type)
```


```python
tuner.fit(paths)
```

### Analyze tuning job results 


```python
# tuning job results dataframe
tuning_job_df = tuner.analytics().dataframe()
tuning_job_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>conv0_channels</th>
      <th>conv0_filter</th>
      <th>conv0_pad</th>
      <th>conv0_pool</th>
      <th>conv0_stride</th>
      <th>conv1_channels</th>
      <th>conv1_filter</th>
      <th>conv1_pad</th>
      <th>conv1_pool</th>
      <th>conv1_stride</th>
      <th>...</th>
      <th>conv2_pool</th>
      <th>conv2_stride</th>
      <th>fc0_neurons</th>
      <th>fc1_neurons</th>
      <th>TrainingJobName</th>
      <th>TrainingJobStatus</th>
      <th>FinalObjectiveValue</th>
      <th>TrainingStartTime</th>
      <th>TrainingEndTime</th>
      <th>TrainingElapsedTimeSeconds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>24.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>64.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>216.0</td>
      <td>267.0</td>
      <td>tensorflow-training-190920-1614-010-2339cf4b</td>
      <td>Completed</td>
      <td>0.9104</td>
      <td>2019-09-20 17:23:39-07:00</td>
      <td>2019-09-20 17:33:02-07:00</td>
      <td>563.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>29.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>54.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>261.0</td>
      <td>218.0</td>
      <td>tensorflow-training-190920-1614-009-61a0d8ce</td>
      <td>Completed</td>
      <td>0.9235</td>
      <td>2019-09-20 17:11:08-07:00</td>
      <td>2019-09-20 17:20:00-07:00</td>
      <td>532.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>26.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>55.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>204.0</td>
      <td>233.0</td>
      <td>tensorflow-training-190920-1614-008-98054b92</td>
      <td>Failed</td>
      <td>NaN</td>
      <td>2019-09-20 17:07:44-07:00</td>
      <td>2019-09-20 17:08:55-07:00</td>
      <td>71.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>26.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>55.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>205.0</td>
      <td>233.0</td>
      <td>tensorflow-training-190920-1614-007-075957e0</td>
      <td>Failed</td>
      <td>NaN</td>
      <td>2019-09-20 17:04:15-07:00</td>
      <td>2019-09-20 17:05:29-07:00</td>
      <td>74.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>27.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>63.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>225.0</td>
      <td>300.0</td>
      <td>tensorflow-training-190920-1614-006-b2bfc6ce</td>
      <td>Failed</td>
      <td>NaN</td>
      <td>2019-09-20 17:00:31-07:00</td>
      <td>2019-09-20 17:01:49-07:00</td>
      <td>78.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>27.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>63.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>224.0</td>
      <td>299.0</td>
      <td>tensorflow-training-190920-1614-005-f7d4ee53</td>
      <td>Failed</td>
      <td>NaN</td>
      <td>2019-09-20 16:56:32-07:00</td>
      <td>2019-09-20 16:58:06-07:00</td>
      <td>94.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>32.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>58.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>253.0</td>
      <td>234.0</td>
      <td>tensorflow-training-190920-1614-004-527c5a6e</td>
      <td>Completed</td>
      <td>0.9057</td>
      <td>2019-09-20 16:50:07-07:00</td>
      <td>2019-09-20 16:54:22-07:00</td>
      <td>255.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>28.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>48.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>249.0</td>
      <td>242.0</td>
      <td>tensorflow-training-190920-1614-003-9198b56d</td>
      <td>Completed</td>
      <td>0.8105</td>
      <td>2019-09-20 16:36:50-07:00</td>
      <td>2019-09-20 16:46:35-07:00</td>
      <td>585.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>27.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>51.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>275.0</td>
      <td>271.0</td>
      <td>tensorflow-training-190920-1614-002-eb3e96e1</td>
      <td>Completed</td>
      <td>0.8946</td>
      <td>2019-09-20 16:27:11-07:00</td>
      <td>2019-09-20 16:33:21-07:00</td>
      <td>370.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>31.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>57.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>214.0</td>
      <td>279.0</td>
      <td>tensorflow-training-190920-1614-001-f2b7ac23</td>
      <td>Completed</td>
      <td>0.8983</td>
      <td>2019-09-20 16:16:17-07:00</td>
      <td>2019-09-20 16:23:07-07:00</td>
      <td>410.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 23 columns</p>
</div>




```python
tuning_job_df['TrainingJobStatus']
```




    0    Completed
    1    Completed
    2       Failed
    3       Failed
    4       Failed
    5       Failed
    6    Completed
    7    Completed
    8    Completed
    9    Completed
    Name: TrainingJobStatus, dtype: object



We note that 4 out of 10 of the jobs failed. After inspecting the [CloudWatch job logs](console.aws.amazon.com/cloudwatch/home), we found that this was due to inappropropriate hyperparameter range choices leading to negative dimension errors. 

This seems especially problematic if Bayesian optimization is driving the search towards incompatible values of the hyperparameters -- training jobs would be more likely to fail and it would be harder to leave a region of hyperparameter space where such failure is likely. Greater care should be taken to avoide incompatible choices of hyperparameters.


```python
tuning_job_df['FinalObjectiveValue'].sort_values(ascending=False)
```




    1    0.9235
    0    0.9104
    6    0.9057
    9    0.8983
    8    0.8946
    7    0.8105
    2       NaN
    3       NaN
    4       NaN
    5       NaN
    Name: FinalObjectiveValue, dtype: float64



Although the validation accuracy improved from job to job, none of the models thus trained achieved a validation accuracy better than the default model,

## Testing

In the end, the default model hyperparameters seemed to be a good option. We'll check the test set performance of the sequence of models learned during [that training job](#Run-a-single-training-job).

As [previously observed](#Analyze-training-history), we expect that weights from this period will perform best on test data, and will be a sound choice for a final model.


```python
! ls models/keras_checkpoints
```

    FashionMNISTCNN-epoch-01-val_acc-0.8778.hdf5
    FashionMNISTCNN-epoch-02-val_acc-0.9033.hdf5
    FashionMNISTCNN-epoch-03-val_acc-0.9091.hdf5
    FashionMNISTCNN-epoch-09-val_acc-0.9281.hdf5
    FashionMNISTCNN-epoch-12-val_acc-0.9281.hdf5
    FashionMNISTCNN-epoch-15-val_acc-0.9306.hdf5
    FashionMNISTCNN-epoch-16-val_acc-0.9310.hdf5
    FashionMNISTCNN-epoch-19-val_acc-0.9339.hdf5
    FashionMNISTCNN-epoch-24-val_acc-0.9367.hdf5
    FashionMNISTCNN-epoch-32-val_acc-0.9386.hdf5
    FashionMNISTCNN-history.csv
    dummy.txt


We'll evaluate all models between epochs 15-32


```python
def epoch_and_val_acc_from_file_name(model_file):
    model_file = model_file.lstrip('FashionMNISTCNN-')
    model_file = model_file.rstrip('.hdf5')
    model_file = model_file.split('-')
    epoch = int(model_file[1])
    val_acc = float(model_file[3])
    return epoch, val_acc

def get_models_from_dir(model_dir, epoch_range, input_shape=(28, 28, 1), drop_rate=0.50):
    models = {}
    for _, _, model_files in os.walk(model_dir):
        for model_file in sorted(model_files):
            if '.hdf5' in model_file:
                epoch, val_acc = epoch_and_val_acc_from_file_name(model_file)
                if epoch in epoch_range:
                    model = fmc(input_shape=input_shape, drop_rate=drop_rate)
                    model.load_weights(os.path.join(model_dir, model_file))
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    models[epoch] = model
    return models

def model_eval_df(models, X, Y):
    losses, accs = [], []
    for epoch in models:
        print("Evaluating epoch {} model:\n".format(epoch))
        loss, acc = models[epoch].evaluate(x=X, y=Y)
        losses += [loss]
        accs += [acc]
    eval_df = pd.DataFrame({'epoch': list(models.keys()), 'test_loss': losses, 'test_acc': accs})
    return eval_df
```


```python
# load and prepare test data
(X_train, Y_train, X_val, Y_val, X_test, Y_test) = fmc.load_data()
(X_train, Y_train, X_val, Y_val, X_test, Y_test) = fmc.prepare_data(X_train, Y_train, X_val, Y_val, X_test, Y_test)
#evaluate models
epoch_range = range(15, 33)
models = get_models_from_dir('models/keras_checkpoints', epoch_range)
model_test_eval_df = model_eval_df(models, X_test, Y_test)
```

    Evaluating epoch 15 model:
    
    10000/10000 [==============================] - 26s 3ms/step
    Evaluating epoch 16 model:
    
    10000/10000 [==============================] - 27s 3ms/step
    Evaluating epoch 19 model:
    
    10000/10000 [==============================] - 25s 3ms/step
    Evaluating epoch 24 model:
    
    10000/10000 [==============================] - 27s 3ms/step
    Evaluating epoch 32 model:
    
    10000/10000 [==============================] - 26s 3ms/step



```python
def plot_performance(model_df):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plt.plot('epoch', 'loss', data=model_df, label='train_loss')
    plt.plot('epoch', 'val_loss', data=model_df, label='val_loss')
    plt.plot('epoch', 'test_loss', data=model_df, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train, val and test loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot('epoch', 'acc', data=model_df, label='train_acc')
    plt.plot('epoch', 'val_acc', data=model_df, label='val_acc')
    plt.plot('epoch', 'test_acc', data=model_df, label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train, val and test accuracy')
    plt.legend()
    
model_df = pd.merge(history_df, model_test_eval_df, on='epoch')
plot_performance(model_df)
```


![png](train_tune_test_files/train_tune_test_43_0.png)



```python
# epochs ranked by test accuracy
model_df.sort_values(by='test_acc', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>val_loss</th>
      <th>val_acc</th>
      <th>loss</th>
      <th>acc</th>
      <th>lr</th>
      <th>epoch</th>
      <th>test_loss</th>
      <th>test_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>0.214470</td>
      <td>0.9339</td>
      <td>0.099285</td>
      <td>0.96342</td>
      <td>0.001</td>
      <td>19</td>
      <td>0.198591</td>
      <td>0.9389</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.231908</td>
      <td>0.9367</td>
      <td>0.076089</td>
      <td>0.97210</td>
      <td>0.001</td>
      <td>24</td>
      <td>0.213586</td>
      <td>0.9386</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.237604</td>
      <td>0.9386</td>
      <td>0.055796</td>
      <td>0.97962</td>
      <td>0.001</td>
      <td>32</td>
      <td>0.220641</td>
      <td>0.9381</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.207615</td>
      <td>0.9310</td>
      <td>0.112607</td>
      <td>0.95914</td>
      <td>0.001</td>
      <td>16</td>
      <td>0.194638</td>
      <td>0.9362</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.200692</td>
      <td>0.9306</td>
      <td>0.122466</td>
      <td>0.95532</td>
      <td>0.001</td>
      <td>15</td>
      <td>0.190690</td>
      <td>0.9322</td>
    </tr>
  </tbody>
</table>
</div>




```python
# epochs ranked by test loss
model_df.sort_values(by='test_loss', ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>val_loss</th>
      <th>val_acc</th>
      <th>loss</th>
      <th>acc</th>
      <th>lr</th>
      <th>epoch</th>
      <th>test_loss</th>
      <th>test_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.200692</td>
      <td>0.9306</td>
      <td>0.122466</td>
      <td>0.95532</td>
      <td>0.001</td>
      <td>15</td>
      <td>0.190690</td>
      <td>0.9322</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.207615</td>
      <td>0.9310</td>
      <td>0.112607</td>
      <td>0.95914</td>
      <td>0.001</td>
      <td>16</td>
      <td>0.194638</td>
      <td>0.9362</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.214470</td>
      <td>0.9339</td>
      <td>0.099285</td>
      <td>0.96342</td>
      <td>0.001</td>
      <td>19</td>
      <td>0.198591</td>
      <td>0.9389</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.231908</td>
      <td>0.9367</td>
      <td>0.076089</td>
      <td>0.97210</td>
      <td>0.001</td>
      <td>24</td>
      <td>0.213586</td>
      <td>0.9386</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.237604</td>
      <td>0.9386</td>
      <td>0.055796</td>
      <td>0.97962</td>
      <td>0.001</td>
      <td>32</td>
      <td>0.220641</td>
      <td>0.9381</td>
    </tr>
  </tbody>
</table>
</div>



As a compromise between test accuracy and loss, we'll select the epoch 19 model for the final model.

## Conclusions

We found that the default model architecture performed well with a test classifiction accuracy of $\approx 93.9\%$ and a categorical cross entropy loss of $\approx 0.199$.

Some possibilities for model improvement are:
- Using data augmentation to increase the size of the training set. This is very easy to implement in Keras
- Better hyperparameter tuning, particularly architecture parameters. This could be done by a more careful definition of the hyperparameter spaces used in [Bayesian tuning](#Analyze-tuning-job-results), or by random search nearby the default hyperparameters
