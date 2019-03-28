import numpy as np
import os


def get_batches(data_dir):
    """
        return a list or generator of (large) ndarrays,
        in order to efficiently utilize GPU
    """
    # todo: read in data that is preoprocessed
    batch_names = os.listdir(data_dir + '/X/')
    n = len(batch_names)
    idx = np.random.permutation(n)
    for i in range(n):
        X = np.load(data_dir + '/X/' + batch_names[idx[i]])
        Y = np.load(data_dir + '/Y/' + batch_names[idx[i]])
        yield X, Y, 32, 0.2
        # X = np.load('train_x_norm_sample.npy')
        # X = np.load('/home/yunhan/data_dir/train_x_224.npy')
        # Y = np.load('train_y_sample.npy')


def get_batches_mono(data_dir):
    """
        return a list or generator of (large) ndarrays,
        in order to efficiently utilize GPU
    """
    X = np.load('/home/yunhan/data_dir/train_x_224.npy')
    # X = np.load('train_x_sample.npy')
    X = X / 255
    # X = np.load('/home/yunhan/data_dir/train_x_224.npy')
    Y = np.load('/home/yunhan/data_dir/train_y_224.npy')
    # Y = np.load('train_y_sample.npy')
    return [(X, Y, 32, 0.2), ]


def get_test_data(data_dir='/home/yunhan/KaggleDiatetic/src/X_3000.npy'):
    X = np.load(data_dir)
    return X
