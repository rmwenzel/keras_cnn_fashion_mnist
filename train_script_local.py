"""Train Keras CNN on Fashion MNIST."""

import argparse
import os

from tensorflow.compat.v1.saved_model import simple_save
from keras import backend as K
from cnn import FashionMNISTCNN


if __name__ == '__main__':

    # parse model parameters from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train', type=str, default='data/')
    parser.add_argument('--val', type=str, default='data/')
    parser.add_argument('--test', type=str, default='data/')
    parser.add_argument('--model', type=str, default='models/')
    parser.add_argument('--checks', type=str, default='models/keras_checkpoints/')
    
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
    train_dir = args.train
    val_dir = args.val
    test_dir = args.test
    model_dir = args.model
    checks_dir = args.checks
    
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

    # fit model
    model.fit(X_train, Y_train, X_val, Y_val,
              checks_dir=checks_dir,
              batch_size=batch_size,
              epochs=epochs)
