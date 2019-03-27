import numpy as np


def get_batches():
    """
        return a list or generator of (large) ndarrays,
        in order to efficiently utilize GPU
    """
    # todo: read in data that is preoprocessed
    # X = np.load('train_x_norm_sample.npy')
    X = np.load('/home/yunhan/data_dir/train_x_norm_224.npy')
    # X = np.load('/home/yunhan/data_dir/train_x_224.npy')
    Y = np.load('/home/yunhan/data_dir/train_y_224.npy')
    # Y = np.load('train_y_sample.npy')

    # naive resampling TODO: better data balencing
    idx = np.concatenate([np.random.choice(np.where(Y==0)[0], 10000),
                          np.random.choice(np.where(Y==1)[0], 10000)])
    return [(X[idx], Y[idx], 32, 0.2), ]
