import sys
import argparse
import numpy as np

# Plug in custom models here
from Inception import InceptionDR

# Plug in data loader
from data_utils import get_batches


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', dest='num_epochs', type=int,
                        default=50, help="Number of epochs to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="learning rate.")

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_epochs = args.num_epochs
    lr = args.lr

    # get model
    model = InceptionDR(model_name='inception_v3',
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        lr=lr)

    # train model
    losses = []
    for epoch in range(num_epochs):
        print("Overall Epoch: %d" % epoch)
        for X, Y, batch_size, valid_split in get_batches():      # get data
            loss = model.train(X, Y, batch_size, valid_split)
            losses.extend(loss)

        if epoch > 0 and epoch % 10 == 0:
            np.savetxt(fname='loss.txt', X=np.array(loss), fmt='%.8lf')

            # save weights periodically
            model.save(epoch)


if __name__ == '__main__':
    main(sys.argv)