# from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
# from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
# from keras import backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
# from PIL import Image
# import os
# import matplotlib.pyplot as plt

def data_parallel_example():
    from keras.utils import multi_gpu_model

    # Replicates `model` on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    parallel_model = multi_gpu_model(model, gpus=3)
    parallel_model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop')

    # This `fit` call will be distributed on 8 GPUs.
    # Since the batch size is 256, each GPU will process 32 samples.
    # parallel_model.fit(x, y, epochs=20, batch_size=256)

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
    for layer in base_model.layers:
        layer.trainable = False

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
parallel_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
X = np.load('train_x_norm_224.npy')
Y = np.load('train_y_224.npy')

hist = parallel_model.fit(
    X, Y,
    epochs=3,
    validation_split=0.2,
    batch_size=96)

model.save('model_freeze.h5')

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
