import torch
import torch.nn as nn
from torch.autograd import Variable

class PoetryNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(PoetryNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)


    def forward(self, input, hidden_state=None):
        seq_len, batch_size = input.size()

        if hidden_state is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden_state

        embeds = self.embeddings(input)
        output, hidden_state = self.lstm(embeds, (h_0, c_0))
        output = self.linear1(output.view(seq_len * batch_size, -1))

        return output, hidden_state

