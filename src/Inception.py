from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.utils import multi_gpu_model
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


class F1Metrics(Callback):
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


class InceptionDR:

    def __init__(self, model_name, optimizer, loss, lr):
        # Wrapping Karasx Inception model
        self.model_name = model_name
        self.lr = lr
        self.optimizer = optimizer
        self.loss_fn = loss

        with tf.device('/cpu:0'):
            input_tensor = Input(shape=(224, 224, 3))

            ###########################################
            # replace this part with model DEFINITION #
            ###########################################
            base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
            self.base_model = base_model

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
        self.model = model

        try:
            parallel_model = multi_gpu_model(model, gpus=3, cpu_merge=False)
            print("using multiple GPUs..")
        except:
            parallel_model = model
            print("using single GPU or CPU..")
        self.parallel_model = parallel_model

        # compile the model (should be done *after* setting layers to non-trainable)
        parallel_model.compile(optimizer=self.optimizer, loss=self.loss_fn)

    def train(self, X, Y, batch_size, valid_split):
        """
        X and Y can be the whole dataset, and also can be a large batch
        ONLY train one iteration over the entire X, Y
        :param X: ndarray
        :param Y: ndarray
        :param batch_size
        :param valid split: batch-wise validation split (for simplicity)
        :return: training losses
        """

        hist = self.parallel_model.fit(
            X, Y,
            epochs=1,
            validation_split=valid_split,
            batch_size=batch_size,
            callbacks=[F1Metrics()])

        return hist.history['loss']

    def save(self, epoch):
        path = '%s_%d.h5' % (self.model_name, epoch)
        self.model.save(path)

    def start_fine_tune(self):
        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(self.base_model.layers):
           print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:249]:
           layer.trainable = False
        for layer in self.model.layers[249:]:
           layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        try:
            parallel_model = multi_gpu_model(self.model, gpus=3, cpu_merge=False)
            print("using multiple GPUs..")
        except:
            parallel_model = self.model
            print("using single GPU or CPU..")
        self.parallel_model = parallel_model

        # compile the model (should be done *after* setting layers to non-trainable)
        from keras.optimizers import SGD
        parallel_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=self.loss_fn)

        # we train our model again (this time fine-tuning the top 2 by calling self.train()
