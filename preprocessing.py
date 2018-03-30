import numpy as np


def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    sample_mean = np.mean(x, axis=1, keepdims=True)
    return x - sample_mean


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    sample_var = np.var(x, axis=1, keepdims=True)
    return scale * x / np.sqrt(bias + sample_var)


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    feature_mean = np.mean(x, axis=0, keepdims=True)
    return x-feature_mean, xtest-feature_mean


def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    n = x.shape[0]
    m = x.shape[1]
    U, S, V = np.linalg.svd(x.T.dot(x/n) + np.eye(m)*bias)
    pca = U.dot(np.diag(1./np.sqrt(S))).dot(U.T)
    return x.dot(pca), xtest.dot(pca)


def kaggle_diabetic_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    x = sample_zero_mean(x)
    xtest = sample_zero_mean(xtest)
    x = gcn(x)
    xtest = gcn(xtest)

    x, xtest = feature_zero_mean(x, xtest)

    x, xtest = zca(x, xtest)

    x = np.reshape(x, [-1, 3, image_size, image_size])
    xtest = np.reshape(xtest, [-1, 3, image_size, image_size])
    return x, xtest
