import sys
import argparse
import numpy as np
import os
# from preprocess import preprocess
# from multiprocessing import Pool
from metrics import calc_metric

# Plug in custom models here
from Inception import InceptionDR

# Plug in data loader
from data_utils import get_evaluate_batches


def validate(model, model_dir, model_name, data_dir):
    """
    :param: model: model object
    :param model_dir: str: dir to model weight file
    :param data_dir: str: path to data
    :param model_name: str: name of model weight file
    :return: validation result for all models
    """
    model_path = model_dir+model_name
    print("loading model weight" + model_path)
    model.load_best_model(model_path)
    print("loading valid data")
    y_pred = []
    y_true = []
    for X, Y in get_evaluate_batches(data_dir):
        _, _, _, y = calc_metric(model, X, Y)
        y_pred.append(y)
        y_true.append(Y)
    pred_data = np.vstack(y_pred)
    true_data = np.vstack(y_true)
    np.save(model_name+'Y_all_pred.npy', pred_data)
    np.save('Y_all_true.npy', true_data)
    return


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dim', dest='input_dim', type=int,
                        default=512, help="Input dimension.")
    parser.add_argument('--output_dim', dest='output_dim', type=int,
                        default=5, help="Output dimension.")
    parser.add_argument('--model_name', dest='model_name', type=str,
                        default="inception_v3_50_50_5class", help="Model name.")
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        default="/home/yunhan/batchified", help="Data dir.")
    parser.add_argument('--model_dir', dest='model_dir', type=str,
                        default="/home/yunhan/KaggleDiatetic/src/inception_v3_5class/", help="Model dir.")
    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    lr = args.lr
    model_name = args.model_name
    input_dim = args.input_dim
    output_dim = args.output_dim
    data_dir = args.data_dir
    model_dir = args.model_dir

    # get model
    print("building model")
    model = InceptionDR(model_name=model_name,
                        input_shape=(input_dim, input_dim, 3),
                        output_dim=output_dim,
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        lr=lr)

    for model_name in os.listdir(model_dir):
        if not model_name.endswith('.h5'):
            continue
        validate(model, model_dir, model_name, data_dir)


if __name__ == '__main__':
    main(sys.argv)
