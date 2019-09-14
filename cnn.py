"""Keras CNN wrapper class."""

import os
import h5py
import numpy as np

from keras.models import Model
from keras.layers import (Input, Dense, Activation,
                          Flatten, BatchNormalization, Conv2D,
                          MaxPooling2D, ZeroPadding2D, Dropout)
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.datasets import fashion_mnist
from keras.utils import to_categorical


class BestValAcc(Callback):
    """Custom callback for logging best validation accuracy."""

    def on_train_begin(self, logs={}):
        self.val_acc = []

    def on_train_end(self, logs={}):
        print("best_val_acc:", max(self.val_acc))

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_acc'))


class CNN(Model):
    """
    CNN with 3 convolutional layers and 2 fully connected hidden layers.

    Subclass of keras.Model

    Parameters
    ---------
    input_shape: tuple
        Shape of image inputs
    conv_params: dict, default {}
        Dictionary of parameters for convolutional layers.
        If empty default values are used.
    fc_params: dict, default {}
        Dictionary of parameters for fully connected layers.
        If empty default values are used.

    Attributes
    -------
    conv_params: dict
        Dictionary of parameters for convolutional layers
    fc_params: dict
        Dictionary of parameters for fully connected layers

    """

    def __init__(self, input_shape, conv_params={}, fc_params={}):
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
                X = Conv2D(c, (f, f), strides=(s, s))(X)
                X = BatchNormalization(name=conv + '_bn')(X)
                X = Activation(act, name=conv + '_act')(X)
                X = MaxPooling2D((o, o), name=conv + '_pool')(X)

        X = Flatten()(X)

        # BN -> FullyConnected blocks
        for (i, fc) in enumerate(fc_params):
            n = fc_params[fc][fc + '_neurons']
            act = fc_params[fc][fc + '_activation']
            X = BatchNormalization(name=fc + '_bn')(X)
            X = Dropout(0.1)(X)
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

    def fit(self, X_train, Y_train, X_val, Y_val, model_dir='models/',
            early_stop_kwargs={}, checkpoint_kwargs={},
            lrreduce_kwargs={}, **kwargs):
        """Wrap fit method with defaults."""
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
                                   save_best_only=True)
        checkpoint_kwargs = {**checkpoint_defaults, **checkpoint_kwargs}
        checkpoint_model_name = ('FashionMNISTCNN-epoch-{epoch:02d}' +
                                 '-val_acc-{val_acc:.4f}.hdf5')
        path = os.path.join(model_dir, checkpoint_model_name)
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
        super().fit(X_train, Y_train, **fit_kwargs,)


class FashionMNISTCNN(CNN):
    """
    Keras CNN with 3 convolutional layers and 2 fully connected hidden layers.

    Subclass of keras.Model

    Parameters
    ---------
    input_shape: tuple
        Shape of image inputs
    conv_params: dict, default {}
        Dictionary of parameters for convolutional layers.
        If empty default values are used.
    fc_params: dict, default {}
        Dictionary of parameters for fully connected layers.
        If empty default values are used.

    Attributes
    -------
    conv_params: dict
        Dictionary of parameters for convolutional layers
    fc_params: dict
        Dictionary of parameters for fully connected layers

    """
    @staticmethod
    def load_data(train_path='data/train.hdf5', valid_path='data/valid.hdf5'):
        """Load MNIST data."""
        # check if data files exist locally
        try:
            with h5py.File(train_path) as hf:
                X_train = np.array(hf['X_train'])
                Y_train = np.array(hf['Y_train'])
            with h5py.File(valid_path) as hf:
                X_val = np.array(hf['X_val'])
                Y_val = np.array(hf['Y_val'])
        # if not get and save locally
        except OSError:
            (X_train, Y_train), (X_val, Y_val) = fashion_mnist.load_data()
            with h5py.File(train_path, 'w') as hf:
                hf.create_dataset('X_train', data=X_train)
                hf.create_dataset('Y_train', data=Y_train)
            with h5py.File(valid_path, 'w') as hf:
                hf.create_dataset('X_val', data=X_val)
                hf.create_dataset('Y_val', data=Y_val)

        return X_train, Y_train, X_val, Y_val

    @staticmethod
    def prepare_data(X_train, Y_train, X_val, Y_val):
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

        return X_train, Y_train, X_val, Y_val
