import math
from torch.optim import Adam
from phoneme_list import *
from dataloaders import *
from model import RNNModule
from utils import *
from conf import *
from warpctc_pytorch import CTCLoss


def train(epoch, model, train_loader, criterion, clip=CLIP, log_interval=LOG_INTV):
    # Turn on training mode which enables dropout.
    model.train()
    optimizer = Adam(model.parameters())
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(BATCH_SIZE)
    for batch, ((packed_x, x_lens), (y_vec, y_lens)) in enumerate(train_loader):
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        logits, hidden = model(packed_x, hidden)
        loss = criterion(logits, y_vec, x_lens, y_lens)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.data

        if (batch+1) % log_interval == 0:
            cur_loss = total_loss[0] / (log_interval*BATCH_SIZE)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                   epoch, batch+1, len(train_loader),
                   elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()


def evaluate(model, valid_loader, criterion, eval_batch_size=EVAL_BSZ):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)
    for (packed_x, x_lens), (y_vec, y_lens) in valid_loader:
        logits, hidden = model(packed_x, hidden)
        loss = criterion(logits, y_vec, x_lens, y_lens)
        total_loss += loss.data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / (len(valid_loader)*EVAL_BSZ)


def train_phase():
    print("Loading data")
    train_x, train_y, valid_x, valid_y = load_data()
    print("train X length = %d" % len(train_x))
    print("train Y length = %d" % len(train_y))
    print("dev X length = %d" % len(valid_x))
    print("dev Y length = %d" % len(valid_y))
    train_loader = MyDataLoader(train_x, train_y)
    valid_loader = MyDataLoader(valid_x, valid_y, batch_size=EVAL_BSZ, evaluation=True)

    print("Building model")
    model = RNNModule(nfreq=FREQ_DIM, nhid=HID_LEN, nphon=PHON_SIZE)

    if torch.cuda.is_available():
        print("Copying to Cuda")
        model.cuda()
        print("Copy finished!")
    criterion = CTCLoss()

    best_val_loss = None
    best_model = None
    print("Start training")
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, MAX_ITER + 1):
            epoch_start_time = time.time()
            train(epoch, model, train_loader, criterion)
            val_loss = evaluate(model, valid_loader, criterion)
            print('-' * 89)
            # import ipdb; ipdb.set_trace()
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_model = get_save_path()
                with open(best_model, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(best_model, 'rb') as f:
        model = torch.load(f)
    return model, criterion


if __name__ == '__main__':
    train_phase()