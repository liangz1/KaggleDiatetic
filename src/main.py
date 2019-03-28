import sys
import argparse
import numpy as np

# Plug in custom models here
from Inception import InceptionDR

# Plug in data loader
from data_utils import get_batches, get_batches_mono


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', dest='num_epochs', type=int,
                        default=50, help="Number of epochs to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="learning rate.")
    parser.add_argument('--input_dim', dest='input_dim', type=int,
                        default=224, help="Input dimension.")
    parser.add_argument('--model_name', dest='model_name', type=str,
                        default="inception_v3_50_50", help="Model name.")
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        default="/home/yunhan/data_dir", help="Data dir.")
    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_epochs = args.num_epochs
    lr = args.lr
    model_name = args.model_name
    input_dim = args.input_dim
    data_dir = args.data_dir

    # get model
    model = InceptionDR(model_name=model_name,
                        input_shape=(input_dim, input_dim, 3),
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        lr=lr)

    # train model
    data = get_batches_mono(data_dir)
    X, Y, batch_size, valid_split = data[0]

    losses = []
    for epoch in range(num_epochs):
        loss = model.train(X, Y, batch_size, valid_split, inner_epoch=1)
        losses.extend(loss)

        np.savetxt(fname='loss.txt', X=np.array(loss), fmt='%.8lf')

        # save weights periodically
        model.save(epoch)

        ##############################
        #print("Overall Epoch: %d" % (epoch+1))
        # for X, Y, batch_size, valid_split in get_batches_mono(data_dir):      # get data
        #     loss = model.train(X, Y, batch_size, valid_split, inner_epoch=num_epochs)
        #     losses.extend(loss)
        #
        #     np.savetxt(fname='loss.txt', X=np.array(loss), fmt='%.8lf')
        #
        #     # save weights periodically
        #     model.save(epoch)


if __name__ == '__main__':
    main(sys.argv)
