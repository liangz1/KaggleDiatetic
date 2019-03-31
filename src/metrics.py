from keras.callbacks import Callback
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
        print("— val_f1: %f — val_precision: %f — val_recall %f" % (f1, precision, recall))
        return


class F1Metrics5Class(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, batch, logs={}):
        if len(self.validation_data) < 2:
            return
        f1, precision, recall, _ = calc_metric(self.model, self.validation_data[0], self.validation_data[1])
        self.val_f1s.append(f1)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        return


def calc_metric(model, X, Y):
    y_pred=np.asarray(model.predict(X))
    predict = y_pred.argmax(axis=1)
    targ = Y
    f1 = f1_score(targ, predict, average=None)
    recall = recall_score(targ, predict, average=None)
    precision = precision_score(targ, predict, average=None)
    for i in range(5):
        print("class %d — val_f1: %f — val_precision: %f — val_recall %f" %
              (i, f1[i], precision[i], recall[i]))
    return f1, precision, recall, y_pred
