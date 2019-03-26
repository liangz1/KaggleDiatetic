# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# from keras import backend as K
#
# def recall(y_true, y_pred):
#     """Recall metric.
#
#     Only computes a batch-wise average of recall.
#
#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
# def precision(y_true, y_pred):
#     """Precision metric.
#
#     Only computes a batch-wise average of precision.
#
#     Computes the precision, a metric for multi-label classification of
#     how many selected items are relevant.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# def f1(y_true, y_pred):
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# from PIL import Image
# import os
# import matplotlib.pyplot as plt

# dir_path = '/home/yunhan/'
# train_path = dir_path+'train'
# valid_path = dir_path+'valid'
# test_path = dir_path+'test'