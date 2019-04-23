import numpy as np
from Inception import InceptionDR
from data_utils import get_test_data_batches, get_test_data_batches_non_normalized


def test_old(best_model_path=
             '/Users/turbo_strong/Desktop/untitled_folder/inception_v3_50_50_13_best_f1.h5'):
    print("building model")
    best_model = InceptionDR("eval")
    print("loading model")
    best_model.load_best_model(best_model_path)
    print("loading test data")
    y_pred = []
    for pix in get_test_data_batches():
        y = best_model.model.predict(pix)
        y_pred.append(y.tolist())
    print('Normalized pred:', y_pred)

    y_pred = []
    for pix in get_test_data_batches_non_normalized():
        y = best_model.model.predict(pix)
        y_pred.append(y.tolist())
    print('Non-normalized pred:', y_pred)
    return

if __name__ == '__main__':
    test_old()

