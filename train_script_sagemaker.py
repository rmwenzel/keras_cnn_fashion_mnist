"""Train Keras CNN on Fashion MNIST on SageMaker."""

import argparse
import os
import h5py
import numpy as np

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
from keras.utils import multi_gpu_model

# Include classes from cnn.py to avoid import issues
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
        except:
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


if __name__ == '__main__':

    # parse model parameters from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int,
                        default=os.environ.get('SM_NUM_GPUS'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--training', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation', type=str,
                        default=os.environ.get('SM_CHANNEL_VALIDATION'))

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
    lr = args.learning_rate
    batch_size = args.batch_size
    gpu_count = args.gpu_count
    
    # if model directory is passed in as hyperparameter use that
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation
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
    model = FashionMNISTCNN(input_shape, conv_params, fc_params)

    print(model.summary())

    # load and prepare data
    train_path = os.path.join(training_dir, 'train.hdf5')
    valid_path = os.path.join(validation_dir, 'valid.hdf5')
    X_train, Y_train, X_val, Y_val = model.load_data(train_path=train_path,
                                                     valid_path=valid_path)
    X_train, Y_train, X_val, Y_val = model.prepare_data(X_train, Y_train,
                                                        X_val, Y_val)

    # compile model with defaults
    model.compile()

    # use multiple gpus if present
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    # fit model
    model.fit(X_train, Y_train, X_val, Y_val,
              model_dir=model_dir,
              batch_size=batch_size,
              epochs=epochs)

    # save Keras model for Tensorflow Serving
    sess = K.get_session()
    simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})
