"""Train Keras CNN on Fashion MNIST."""

import argparse
import os
import h5py
import numpy as np

from tensorflow import logging
from tensorflow.compat.v1.saved_model import simple_save

from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Dense, Activation,
                          Flatten, BatchNormalization, Conv2D,
                          MaxPooling2D, ZeroPadding2D)
from keras.utils import to_categorical
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)


def FashionMNISTModel(input_shape, conv_params, fc_params):
    """
    Keras model for Fashion MNIST data.

    Parameters
    ---------
    input_shape: tuple
        Shape of image inputs

    Returns
    -------
    model: keras.Model()
        Compiled model for image data, not yet fit.

    """
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
        X = Dense(n, activation=act, name=fc + '_act')(X)

    # create model
    model = Model(input=X_input, outputs=X, name='FashionMNISTModel')

    return model


class BestValAcc(Callback):
    """Custom callback for logging best validation accuracy."""

    def on_train_begin(self, logs={}):
        self.val_acc = []

    def on_train_end(self, logs={}):
        print("best_val_acc:", max(self.val_acc))

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_acc'))


if __name__ == '__main__':

    # supress tf backend FutureWarnings
    logging.set_verbosity(logging.ERROR)

    # parse model parameters from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)

    # architecture hyperparameters from sagemaker.tuner.HyperparameterTuner
    # called in notebook keras_cnn_fashion_mnist.ipynb
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
    lr = args.learning_rate
    batch_size = args.batch_size
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

    training_dir = validation_dir = 'data/'
    model_dir = 'models/'

    # read in train and validation data
    with h5py.File(os.path.join(training_dir, 'train.hdf5'), 'r') as hf:
        X_train = np.array(hf['X_train'])
        Y_train = np.array(hf['Y_train'])

    with h5py.File(os.path.join(validation_dir, 'val.hdf5'), 'r') as hf:
        X_val = np.array(hf['X_val'])
        Y_val = np.array(hf['Y_val'])

    # reshape for keras
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Normalize pixel values
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_val /= 255

    # One-hot encode response
    Y_train = to_categorical(Y_train, 10)
    Y_val = to_categorical(Y_val, 10)

    # collect layer parameters
    conv_params = {'conv0': conv0_params, 'conv1': conv1_params,
                   'conv2': conv2_params}
    fc_params = {'fc0': fc0_params, 'fc1': fc1_params, 'fc2': fc2_params}

    model = FashionMNISTModel(input_shape, conv_params, fc_params)

    print(model.summary())

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callback for early stopping
    early_stopping = EarlyStopping(monitor='val_acc',
                                   min_delta=0,
                                   patience=20,
                                   verbose=1, mode='auto')

    # Custom callback for best validation accuracy for automatic model tuning
    best_val_acc = BestValAcc()

    # Define callback to save best epoch
    checkpointer = ModelCheckpoint(os.path.join(os.getcwd()
                                   + 'checkpoints/keras-model.hdf5'),
                                   monitor='val_acc', verbose=1,
                                   save_best_only=True)
    # Reduce learning rate if accuracy plateaus
    lrreduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                 patience=10, verbose=1)

    callbacks = [early_stopping, best_val_acc, checkpointer, lrreduce]

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              validation_data=(X_val, Y_val),
              epochs=epochs,
              verbose=1,
              callbacks=callbacks)

    # save Keras model for Tensorflow Serving
    sess = K.get_session()
    simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})
