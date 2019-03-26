import numpy as np


def get_batches():
    """
        return a list or generator of (large) ndarrays,
        in order to efficiently utilize GPU
    """
    # todo: read in data that is preoprocessed
    X = np.load('/home/yunhan/data_dir/train_x_norm_224.npy')
    X = np.load('/home/yunhan/data_dir/train_x_224.npy')
    Y = np.load('/home/yunhan/data_dir/train_y_224.npy')

    # naive resampling TODO: better data balencing
    idx = np.concatenate([np.random.choice(np.where(Y==0)[0], 17560),
                          np.random.choice(np.where(Y==1)[0], 17560)])
    return [(X[idx], Y[idx])]