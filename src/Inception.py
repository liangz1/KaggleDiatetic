from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.utils import multi_gpu_model
import tensorflow as tf
from metrics import F1Metrics5Class


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

            # add a global spatial average pooling layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            # let's add a fully-connected layer

            x = Dense(50, activation='relu')(x)
            # x = Dropout(0.5)(x)
            x = Dense(50, activation='relu')(x)
            # x = Dropout(0.5)(x)

            # and a logistic layer -- we have output_dim classes
            predictions = Dense(output_dim, activation='softmax')(x)

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
        parallel_model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

    def train(self, X, Y, valid_data=None, batch_size=32, inner_epoch=1):
        """
        X and Y can be the whole dataset, and also can be a large batch
        ONLY train one iteration over the entire X, Y
        :param X: ndarray
        :param Y: ndarray
        :param batch_size
        :param valid split: batch-wise validation split (for simplicity)
        :return: training losses
        """
        class_weight = {0: 1, 2: 5, 1: 10.5, 4: 34.1, 3: 31.3}
        print("class weight %s" % str(class_weight))
        hist = self.parallel_model.fit(
            X, Y,
            epochs=inner_epoch,
            shuffle=True,
            class_weight=class_weight,
            validation_data=valid_data,
            batch_size=batch_size,
            callbacks=[F1Metrics5Class()])

        return hist.history['loss']

    def validate(self, X, Y):
        """
        X and Y can be the whole dataset, and also can be a large batch
        ONLY train one iteration over the entire X, Y
        :param X: ndarray
        :param Y: ndarray
        :param batch_size
        :param valid split: batch-wise validation split (for simplicity)
        :return: training losses
        """

        hist = self.parallel_model.evaluate(
            X, Y, callbacks=[F1Metrics5Class()])

        return hist.history

    def save(self, epoch):
        path = '%s_%d.h5' % (self.model_name, epoch)
        self.model.save_weights(path)

    def load_best_model(self, path='best_model.h5'):
        self.model.load_weights(path)

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
