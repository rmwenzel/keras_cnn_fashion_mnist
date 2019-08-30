"""Train Keras CNN on Fashion MNIST."""

import argparse
import os
import h5py
import numpy as np

from tensorflow import saved_model, logging
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Dense, Activation,
                          Flatten, BatchNormalization, Conv2D,
                          MaxPooling2D, ZeroPadding2D)
from keras.utils import multi_gpu_model, to_categorical
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
        p = conv_params[conv]['pad']
        c = conv_params[conv]['channels']
        f = conv_params[conv]['filter']
        s = conv_params[conv]['stride']
        o = conv_params[conv]['pool']
        act = conv_params[conv]['activation']
        if i == 0:
            X = ZeroPadding2D((p, p))(X_input)
            X = Conv2D(c, (f, f), strides=(s, s))(X)
            X = BatchNormalization()(X)
            X = Activation(act)(X)
            X = MaxPooling2D((o, o))(X)
        else:
            X = ZeroPadding2D((p, p))(X)
            X = Conv2D(c, (f, f), strides=(s, s))(X)
            X = BatchNormalization()(X)
            X = Activation(act)(X)
            X = MaxPooling2D((o, o))(X)

    X = Flatten()(X)

    # BN -> FullyConnected blocks
    for (i, fc) in enumerate(fc_params):
        n = fc_params[fc]['neurons']
        act = fc_params[fc]['activation']
        X = BatchNormalization()(X)
        X = Dense(n, activation=act, name=fc)(X)

    # create model
    model = Model(input=X_input, outputs=X, name='FashionMNISTModel')

    return model


class BestValAcc(Callback):
    """Custom callback for logging best validation accuracy."""

    def on_train_begin(self, logs={}):
        self.val_acc = []

    def on_train_end(self, logs={}):
        print("Best val_acc:", max(self.val_acc))

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_acc'))


if __name__ == '__main__':

    logging.set_verbosity(logging.ERROR)

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int,
                        default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str,
                        default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation

    with h5py.File(os.path.join(training_dir, 'train.hdf5'), 'r') as hf:
        X_train = np.array(hf['X_train'])
        Y_train = np.array(hf['Y_train'])

    with h5py.File(os.path.join(validation_dir, 'train.hdf5'), 'r') as hf:
        X_val = np.array(hf['X_train'])
        Y_val = np.array(hf['Y_train'])

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Normalize pixel values
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_val /= 255

    # Convert class vectors to binary class matrices
    Y_train = to_categorical(Y_train, 10)
    Y_val = to_categorical(Y_val, 10)

    # layer parameters
    conv0_params = {'pad': 1, 'channels': 32, 'filter': 3,
                    'stride': 1, 'pool': 1, 'activation': 'relu'}
    conv1_params = {'pad': 1, 'channels': 64, 'filter': 3,
                    'stride': 1, 'pool': 2, 'activation': 'relu'}
    conv2_params = {'pad': 1, 'channels': 128, 'filter': 3,
                    'stride': 1, 'pool': 2, 'activation': 'relu'}
    conv_params = {'conv0': conv0_params, 'conv1': conv1_params,
                   'conv2': conv2_params}
    fc0_params = {'neurons': 256, 'activation': 'relu'}
    fc1_params = {'neurons': 512, 'activation': 'relu'}
    fc2_params = {'neurons': 10, 'activation': 'softmax'}
    fc_params = {'fc0': fc0_params, 'fc1': fc1_params, 'fc2': fc2_params}

    model = FashionMNISTModel(input_shape, conv_params, fc_params)

    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

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
    checkpointer = ModelCheckpoint(filepath=(model_dir+'/'
                                   + 'fashion-mnist-model.hdf5'),
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
    saved_model.save(sess, os.path.join(model_dir, 'model/1'),
                     inputs={'inputs': model.input},
                     outputs={t.name: t for t in model.outputs})
