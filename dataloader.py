from torch.utils.data.dataloader import DataLoader
import numpy as np
from conf import *
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


# reference: https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e

def my_collate_fn(batch: np.ndarray):
    batch_size = len(batch)

    if torch.cuda.is_available():
        seq_lengths = torch.cuda.IntTensor(list(map(lambda x: len(x[0]), batch)))
        phon_lengths = torch.cuda.IntTensor(list(map(lambda x: len(x[1]), batch)))
        seq_tensor = Variable(torch.zeros((batch_size, seq_lengths.max(), FREQ_DIM))).cuda()
        for idx, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
            # import ipdb; ipdb.set_trace()
            seq_tensor[idx, :seqlen] = torch.cuda.FloatTensor(seq[0])
    else:
        seq_lengths = torch.IntTensor(list(map(lambda x: len(x[0]), batch)))
        phon_lengths = torch.IntTensor(list(map(lambda x: len(x[1]), batch)))
        seq_tensor = Variable(torch.zeros((batch_size, seq_lengths.max(), FREQ_DIM)))
        for idx, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.FloatTensor(seq[0])
    # sort the sequences by length descent
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    phon_lengths = phon_lengths[perm_idx]
    phon_array = [torch.IntTensor(batch[perm_idx[i]][1]+1) for i in range(batch_size)]
    phon_array = Variable(torch.cat(phon_array))
    seq_tensor = seq_tensor.transpose(0, 1)
    packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
    # import ipdb; ipdb.set_trace()
    return (packed_input, Variable(seq_lengths.cpu())), (phon_array, Variable(phon_lengths.cpu()))


class MyDataLoader(DataLoader):
    def __init__(self, data_x, data_y, batch_size=BATCH_SIZE, evaluation=False):
        dataset = list(zip(data_x, data_y))
        super(MyDataLoader, self).__init__(dataset, batch_size=batch_size,
                                           shuffle=(not evaluation),
                                           collate_fn=my_collate_fn,
                                           drop_last=True)
        self.evaluation = evaluation
        # fixed seq length
        self.N = data_x.shape[0]
        # ignore the last incomplete batch
        self.nbatch = self.N // batch_size

    def __len__(self):
        return self.nbatch


def my_test_collate_fn(batch: np.ndarray):
    batch_size = len(batch)

    if torch.cuda.is_available():
        seq_lengths = torch.cuda.IntTensor(list(map(len, batch)))
        seq_tensor = Variable(torch.zeros((batch_size, seq_lengths.max(), FREQ_DIM))).cuda()
        for idx, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
            # import ipdb; ipdb.set_trace()
            seq_tensor[idx, :seqlen] = torch.cuda.FloatTensor(seq)
    else:
        seq_lengths = torch.IntTensor(list(map(len, batch)))
        phon_lengths = torch.IntTensor(list(map(len, batch)))
        seq_tensor = Variable(torch.zeros((batch_size, seq_lengths.max(), FREQ_DIM)))
        for idx, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.FloatTensor(seq)
    seq_tensor = seq_tensor.transpose(0, 1)
    packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
    # import ipdb; ipdb.set_trace()
    return packed_input, seq_lengths.cpu()


class MyTestDataLoader(DataLoader):
    def __init__(self, data_x, batch_size=1, evaluation=True):
        super(MyTestDataLoader, self).__init__(data_x, batch_size=batch_size,
                                               shuffle=(not evaluation),
                                               collate_fn=my_test_collate_fn)
        self.evaluation = evaluation
        # fixed seq length
        self.N = data_x.shape[0]
        # ignore the last incomplete batch
        self.nbatch = self.N // batch_size

    def __len__(self):
        return self.nbatch
