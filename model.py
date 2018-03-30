from torch.autograd import Variable
from torch.nn import *
from torch.nn.utils.rnn import pad_packed_sequence


class RNNModule(Module):
    """Container module with an encoder, 3 LSTMs, and a decoder.
    """
    def __init__(self, nfreq=40, nhid=256, nlayers=3, nphon=47, dropout=0.5):
        """

        :param nfreq: number of frequency features = 40
        :param nhid: number of hidden units per layer
        :param nlayers: number of layers
        :param nphon: number of phonemes, including a blank (output shape)
        :param dropout: remaining portion
        """
        super(RNNModule, self).__init__()
        self.drop = Dropout(dropout)
        self.rnn = LSTM(nfreq, nhid, nlayers, dropout=dropout, bidirectional=True)
        self.decoder = Linear(nhid*2, nphon)
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        packed_output, hidden = self.rnn(input, hidden)
        output = pad_packed_sequence(packed_output)
        # import ipdb; ipdb.set_trace()
        output = self.drop(output[0])
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers*2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers*2, bsz, self.nhid).zero_()))
