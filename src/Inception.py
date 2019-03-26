# from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
# from keras.preprocessing import image
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras import backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# from PIL import Image
# import os
# import matplotlib.pyplot as plt

# dir_path = '/home/yunhan/'
# train_path = dir_path+'train'
# valid_path = dir_path+'valid'
# test_path = dir_path+'test'

# create the base pre-trained model
with tf.device('/cpu:0'):
    input_tensor = Input(shape=(224, 224, 3))
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(2, activation='softmax')(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

# try:
#     parallel_model = multi_gpu_model(model, gpus=3, cpu_merge=False)
#     print("Training using multiple GPUs..")
# except:
#     parallel_model = model
#     print("Training using single GPU or CPU..")

parallel_model = multi_gpu_model(model, gpus=3)

# compile the model (should be done *after* setting layers to non-trainable)
parallel_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# train the model on the new data for a few epochs
X = np.load('train_x_norm_224.npy')
X = np.load('train_x_224.npy')
Y = np.load('train_y_224.npy')
idx = np.concatenate([np.random.choice(np.where(Y==0)[0], 17560), np.random.choice(np.where(Y==1)[0], 17560)])
X_balance = X[idx]
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

class Metrics(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0])).argmax(axis=1)
        targ = self.validation_data[1]
        f1 = f1_score(targ, predict)
        recall = recall_score(targ, predict)
        precision = precision_score(targ, predict)
        self.val_f1s.append(f1)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" %(f1, precision, recall))
        return

metrics = Metrics()

hist = parallel_model.fit(
    X, Y,
    epochs=50,
    validation_split=0.2,
    batch_size=96,
    callbacks=[metrics])

model.save('model.h5')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)
#
# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True
#
# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2
