from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.utils import multi_gpu_model
import tensorflow as tf


class InceptionDR:

    def __init__(self, model_name='my_model',
                 input_shape=(224, 224, 3),
                 output_dim=2,
                 optimizer='sgd',
                 loss='sparse_categorical_crossentropy',
                 lr=0.0001):
        # Wrapping Karas Inception model
        self.model_name = model_name
        self.output_dim = output_dim
        self.lr = lr
        self.optimizer = optimizer
        self.loss_fn = loss

        with tf.device('/cpu:0'):
            input_tensor = Input(shape=input_shape)
            ###########################################
            # replace this part with model DEFINITION #
            ###########################################
            base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
            self.base_model = base_model
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(50, activation='relu')(x)
            x = Dense(50, activation='relu')(x)
            predictions = Dense(output_dim, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        self.model = model

        try:
            parallel_model = multi_gpu_model(model, gpus=3, cpu_merge=False)
            print("using multiple GPUs..")
        except:
            parallel_model = model
            print("using single GPU or CPU..")
        self.parallel_model = parallel_model

        # compile the model (should be done *after* setting layers to non-trainable)
        parallel_model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

    def load_best_model(self, path='best_model.h5'):
        self.model.load_weights(path)
