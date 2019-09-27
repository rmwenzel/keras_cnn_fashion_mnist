"""Train Keras CNN on Fashion MNIST on SageMaker."""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import boto3

from tensorflow.compat.v1.saved_model import simple_save
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Dense, Activation,
                          Flatten, BatchNormalization, Conv2D,
                          MaxPooling2D, ZeroPadding2D, Dropout)
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.datasets import fashion_mnist
from keras.utils import to_categorical, multi_gpu_model



# Include classes from cnn.py to avoid import issues
class BestValAcc(Callback):

    def on_train_begin(self, logs={}):
        self.val_acc = []

    def on_train_end(self, logs={}):
        print("best_val_acc:", max(self.val_acc))

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_acc'))

        
class CNN(Model):

    def __init__(self, input_shape, conv_params={}, fc_params={}, drop_rate=0.0):
        # param defaults
        conv0_defaults = {'conv0_pad': 1,
                          'conv0_channels': 32,
                          'conv0_filter': 3,
                          'conv0_stride': 1,
                          'conv0_pool': 1,
                          'conv0_activation': 'relu'}
        conv1_defaults = {'conv1_pad': 1,
                          'conv1_channels': 64,
                          'conv1_filter': 3,
                          'conv1_stride': 1,
                          'conv1_pool': 2,
                          'conv1_activation': 'relu'}
        conv2_defaults = {'conv2_pad': 1,
                          'conv2_channels': 128,
                          'conv2_filter': 3,
                          'conv2_stride': 1,
                          'conv2_pool': 2,
                          'conv2_activation': 'relu'}
        fc0_defaults = {'fc0_neurons': 512,
                        'fc0_activation': 'relu'}
        fc1_defaults = {'fc1_neurons': 256,
                        'fc1_activation': 'relu'}
        fc2_defaults = {'fc2_neurons': 10,
                        'fc2_activation': 'softmax'}

        conv_defaults = {'conv0': conv0_defaults,
                         'conv1': conv1_defaults,
                         'conv2': conv2_defaults}
        fc_defaults = {'fc0': fc0_defaults,
                       'fc1': fc1_defaults,
                       'fc2': fc2_defaults}

        # set param attributes
        self.conv_params = conv_params
        self.fc_params = fc_params

        # merge passed in params with defaults
        for layer in conv_defaults:
            try:
                conv_params[layer] = {**conv_defaults[layer],
                                      **conv_params[layer]}
            except KeyError:
                conv_params[layer] = conv_defaults[layer]
        for layer in fc_defaults:
            try:
                fc_params[layer] = {**fc_params[layer],
                                    **fc_defaults[layer]}
            except KeyError:
                fc_params[layer] = fc_defaults[layer]

        # Input placeholder
        X_input = Input(input_shape)

        # Pad -> Conv -> Act -> BN -> MaxPool blocks
        for (i, conv) in enumerate(conv_params):
            p = conv_params[conv][conv + '_pad']
            c = conv_params[conv][conv + '_channels']
            f = conv_params[conv][conv + '_filter']
            s = conv_params[conv][conv + '_stride']
            o = conv_params[conv][conv + '_pool']
            act = conv_params[conv][conv + '_activation']
            if i == 0:
                X = ZeroPadding2D((p, p), name=conv + '_pad')(X_input)
                X = Conv2D(c, (f, f), strides=(s, s), name=conv)(X)
                X = BatchNormalization(name=conv + '_bn')(X)
                X = Activation(act, name=conv + '_act')(X)
                X = MaxPooling2D((o, o), name=conv + '_pool')(X)
            else:
                X = ZeroPadding2D((p, p), name=conv + '_pad')(X)
                X = Conv2D(c, (f, f), strides=(s, s), name=conv)(X)
                X = BatchNormalization(name=conv + '_bn')(X)
                X = Activation(act, name=conv + '_act')(X)
                X = MaxPooling2D((o, o), name=conv + '_pool')(X)

        X = Flatten()(X)

        # BN -> FullyConnected blocks
        for (i, fc) in enumerate(fc_params):
            n = fc_params[fc][fc + '_neurons']
            act = fc_params[fc][fc + '_activation']
            X = BatchNormalization(name=fc + '_bn')(X)
            X = Dropout(drop_rate, name=fc + '_drop')(X)
            X = Dense(n, activation=act, name=fc + '_act')(X)

        # create model
        super().__init__(outputs=X, inputs=X_input)

        # set param attributes
        self.conv_params = conv_params
        self.fc_params = fc_params

    def compile(self, **kwargs):
        """Wrap compile method with defaults."""
        defaults = dict(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        new_kwargs = {**defaults, **kwargs}
        super().compile(**new_kwargs)

    def fit(self, X_train, Y_train, X_val, Y_val, 
            checks_dir='models/keras_checkpoints',
            early_stop_kwargs={}, checkpoint_kwargs={},
            lrreduce_kwargs={}, **kwargs):
            """Wrap fit method with defaults.

            Parameters
            ----------
            X_train: numpy.ndarray
                Array of training data inputs
            Y_train: numpy.ndarray
                Array of training data outputs
            X_val: numpy.ndarray
                Array of validation data inputs
            Y_val: numpy.ndarray
                Array of validation data outputs
            checks_dir: str, default 'models/'
                Path to directory for saving checkpoints
            early_stop_kwargs: dict, default empty
                Keyword arguments for early stopping callback
            checkpoint_stop_kwargs: dict, default empty
                Keyword arguments for checkpoint callback
            lrreduce_kwargs: dict, default empty
                Keyword arguments for reduce learning rate on plateau callback
            **kwargs:
                Keyword arguments for keras.Model.fit

            """
            # Stop training if validation accuracy doesn't improve
            early_stop_defaults = dict(monitor='val_acc',
                                       min_delta=0,
                                       patience=10,
                                       verbose=1,
                                       mode='auto')
            early_stop_kwargs = {**early_stop_defaults, **early_stop_kwargs}
            early_stopping = EarlyStopping(**early_stop_kwargs)

            # Save if validation accuracy improves
            checkpoint_defaults = dict(monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)
            checkpoint_kwargs = {**checkpoint_defaults, **checkpoint_kwargs}
            checkpoint_model_name = ('FashionMNISTCNN-epoch-{epoch:02d}' +
                                     '-val_acc-{val_acc:.4f}.hdf5')
            path = os.path.join(checks_dir, checkpoint_model_name)
            checkpointer = ModelCheckpoint(path, **checkpoint_kwargs)

            # Reduce learning rate if accuracy plateaus
            lrreduce_defaults = dict(monitor='val_acc',
                                     factor=0.1,
                                     patience=10,
                                     verbose=1)
            lrreduce_kwargs = {**lrreduce_defaults, **lrreduce_kwargs}
            lrreduce = ReduceLROnPlateau(**lrreduce_kwargs)

            # Track best validation accuracy
            best_val_acc = BestValAcc()

            callbacks = [early_stopping, best_val_acc, checkpointer,
                         lrreduce]

            fit_defaults = dict(batch_size=128,
                                validation_data=(X_val, Y_val),
                                epochs=1,
                                verbose=1,
                                callbacks=callbacks)
            fit_kwargs = {**fit_defaults, **kwargs}
            history = super().fit(X_train, Y_train, **fit_kwargs)
            return history


class FashionMNISTCNN(CNN):
    
    @staticmethod
    def _create_test_set(X_train, Y_train, test_size=10000, seed=27):
        # random seed for reproducibility
        np.random.seed(seed)
        # create dataframe for convenience
        train_df = pd.DataFrame(X_train.reshape(X_train.shape[0], 784))
        train_df['label'] = Y_train
        # store slices for later concatenation     
        slices = []
        # get slices for all the classes
        for class_label in train_df['label'].unique():
            # slice all rows for this class
            class_slice = train_df[train_df['label'] == class_label]
            # get indices for test rows
            indices = np.random.choice(class_slice.index.values, 
                                             size=test_size//10, 
                                             replace=False)
            # slice for these indices
            slices += [class_slice.loc[indices, : ]]
            # drop rows for these indices 
            train_df = train_df.drop(index=indices)
        # collect slices into a dataframe
        test_df = pd.concat(slices, ignore_index=True)
        # convert back to numpy arrays
        X_train = train_df.drop(columns=['label']).values
        Y_train = train_df['label'].values
        X_test = test_df.drop(columns=['label']).values
        Y_test = test_df['label'].values
        # reshape inputs
        X_train = X_train.reshape(X_train.shape[0], 28, 28)
        X_test = X_test.reshape(Y_test.shape[0], 28, 28)
        # return numpy arrays of values
        return X_train, Y_train, X_test, Y_test
    
    @staticmethod
    def load_data(train_path='data/train.hdf5', val_path='data/val.hdf5',
                  test_path='data/test.hdf5'):
        """Load Fashion MNIST data."""
        # check if data files exist locally
        try:
            with h5py.File(train_path) as hf:
                X_train = np.array(hf['X_train'])
                Y_train = np.array(hf['Y_train'])
            with h5py.File(val_path) as hf:
                X_val = np.array(hf['X_val'])
                Y_val = np.array(hf['Y_val'])
            with h5py.File(test_path) as hf:
                X_test = np.array(hf['X_test'])
                Y_test = np.array(hf['Y_test'])
                
        # if not get and save locally
        except:
            (X_train, Y_train), (X_val, Y_val) = fashion_mnist.load_data()
            X_train, Y_train, X_test, Y_test = FashionMNISTCNN._create_test_set(X_train, 
                                                                                Y_train)
            with h5py.File(train_path, 'w') as hf:
                hf.create_dataset('X_train', data=X_train)
                hf.create_dataset('Y_train', data=Y_train)
            with h5py.File(val_path, 'w') as hf:
                hf.create_dataset('X_val', data=X_val)
                hf.create_dataset('Y_val', data=Y_val)
            with h5py.File(test_path, 'w') as hf:
                hf.create_dataset('X_test', data=X_test)
                hf.create_dataset('Y_test', data=Y_test)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test
        

    @staticmethod
    def prepare_data(X_train, Y_train, X_val, Y_val, X_test=None, Y_test=None):
        """Prepare data for model."""
        # reshape for keras
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)

        # Normalize pixel values
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        X_train /= 255
        X_val /= 255
        
        # One-hot encode image classes
        Y_train = to_categorical(Y_train, 10)
        Y_val = to_categorical(Y_val, 10)
        
        if X_test is not None:
            X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
            X_test = X_test.astype('float32')
            X_test /= 255
            
        if Y_test is not None:
            Y_test = to_categorical(Y_test, 10)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    @staticmethod
    def upload_checks_to_s3(checks_output_path, checks_dir):
        """Put keras checkpoints in outside s3 bucket"""
        s3_resource = boto3.resource('s3')
        bucket_name = os.path.dirname(checks_output_path).split('//')[1]
        prefix = os.path.basename(checks_output_path)
        bucket = s3_resource.Bucket(bucket_name)

        for _, _, files in os.walk(checks_dir):
            for file in files:
                file_path = os.path.join(checks_dir, file)
                with open(file_path, 'rb') as data:
                    bucket.put_object(Key=os.path.join(prefix, file), Body=data)

    @staticmethod
    def save_history(history, checks_dir):
        """Save keras history in checkpoints directory"""
        # convert the history.history dict to a pandas DataFrame:     
        history_df = pd.DataFrame(history.history) 
        history_df['epoch'] = history_df.index + 1
        # or save to csv: 
        history_csv_file = 'FashionMNISTCNN-history.csv'
        path = os.path.join(checks_dir, history_csv_file)
        with open(path, mode='w') as f:
            history_df.to_csv(f, index=False)
    
if __name__ == '__main__':

    # parse model parameters from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--drop-rate', type=float, default=0.0)
    parser.add_argument('--checks-out-path', type=str, 
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--gpu-count', type=int,
                        default=os.environ.get('SM_NUM_GPUS'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str,
                        default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--test', type=str,
                        default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--checks', type=str,
                        default=os.environ.get('SM_CHANNEL_CHECKS'))

    # architecture hyperparameters
    parser.add_argument('--conv0_pad', type=int, default=1)
    parser.add_argument('--conv0_channels', type=int, default=32)
    parser.add_argument('--conv0_filter', type=int, default=3)
    parser.add_argument('--conv0_stride', type=int, default=1)
    parser.add_argument('--conv0_pool', type=int, default=1)
    parser.add_argument('--conv0_activation', type=str, default='relu')

    parser.add_argument('--conv1_pad', type=int, default=1)
    parser.add_argument('--conv1_channels', type=int, default=64)
    parser.add_argument('--conv1_filter', type=int, default=3)
    parser.add_argument('--conv1_stride', type=int, default=1)
    parser.add_argument('--conv1_pool', type=int, default=2)
    parser.add_argument('--conv1_activation', type=str, default='relu')

    parser.add_argument('--conv2_pad', type=int, default=1)
    parser.add_argument('--conv2_channels', type=int, default=128)
    parser.add_argument('--conv2_filter', type=int, default=3)
    parser.add_argument('--conv2_stride', type=int, default=1)
    parser.add_argument('--conv2_pool', type=int, default=2)
    parser.add_argument('--conv2_activation', type=str, default='relu')

    parser.add_argument('--fc0_neurons', type=int, default=512)
    parser.add_argument('--fc0_activation', type=str, default='relu')
    parser.add_argument('--fc1_neurons', type=int, default=256)
    parser.add_argument('--fc1_activation', type=str, default='relu')

    # store parameters
    args, _ = parser.parse_known_args()

    epochs = args.epochs
    batch_size = args.batch_size
    drop_rate = args.drop_rate
    gpu_count = args.gpu_count
    model_dir = args.model
    train_dir = args.train
    val_dir = args.val
    test_dir = args.test
    checks_dir = args.checks
    checks_out_path = args.checks_out_path
    
    conv0_params = {'conv0_pad': args.conv0_pad,
                    'conv0_channels': args.conv0_channels,
                    'conv0_filter': args.conv0_filter,
                    'conv0_stride': args.conv0_stride,
                    'conv0_pool': args.conv0_pool,
                    'conv0_activation': args.conv0_activation}
    conv1_params = {'conv1_pad': args.conv1_pad,
                    'conv1_channels': args.conv1_channels,
                    'conv1_filter': args.conv1_filter,
                    'conv1_stride': args.conv1_stride,
                    'conv1_pool': args.conv1_pool,
                    'conv1_activation': args.conv1_activation}
    conv2_params = {'conv2_pad': args.conv2_pad,
                    'conv2_channels': args.conv2_channels,
                    'conv2_filter': args.conv2_filter,
                    'conv2_stride': args.conv2_stride,
                    'conv2_pool': args.conv2_pool,
                    'conv2_activation': args.conv2_activation}
    fc0_params = {'fc0_neurons': args.fc0_neurons,
                  'fc0_activation': args.fc0_activation}
    fc1_params = {'fc1_neurons': args.fc1_neurons,
                  'fc1_activation': args.fc1_activation}
    fc2_params = {'fc2_neurons': 10,
                  'fc2_activation': 'softmax'}

    # collect layer parameters
    conv_params = {'conv0': conv0_params, 'conv1': conv1_params,
                   'conv2': conv2_params}
    fc_params = {'fc0': fc0_params, 'fc1': fc1_params, 'fc2': fc2_params}

    # create model
    input_shape = (28, 28, 1)
    model = FashionMNISTCNN(input_shape, conv_params, fc_params, drop_rate)

    print(model.summary())

    # load and prepare data
    train_path = os.path.join(train_dir, 'train.hdf5')
    val_path = os.path.join(val_dir, 'val.hdf5')
    test_path = os.path.join(test_dir, 'test.hdf5')

    X_train, Y_train, X_val, Y_val, _, _ = model.load_data(train_path=train_path,
                                                           val_path=val_path,
                                                           test_path=test_path)
    X_train, Y_train, X_val, Y_val, _, _ = model.prepare_data(X_train, Y_train,
                                                              X_val, Y_val)

    # compile model with defaults
    model.compile()

    # use multiple gpus if present
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    # fit model
    history = model.fit(X_train, Y_train, X_val, Y_val,
              checks_dir=checks_dir,
              batch_size=batch_size,
              epochs=epochs)

    # upload Keras checkpoints and history to s3
    model.save_history(history, checks_dir)
    model.upload_checks_to_s3(checks_out_path, checks_dir)
    
    # save Keras model for Tensorflow Serving
    sess = K.get_session()
    simple_save(sess,
                os.path.join(model_dir, 'model/1'),
                inputs={'inputs': model.input},
                outputs={t.name: t for t in model.outputs})
