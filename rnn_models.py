import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# RNN class that passes forward the hidden unit at every step
class RNN1(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNN1, self).__init__()
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = nn.Linear(input_size + hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, input_layer, hidden):
        combined = torch.cat((input_layer, hidden), 1)
        hidden = self.input2hidden(combined)
        output = self.input2output(combined)
        return hidden, output

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    
# RNN class with nonlinearity between hidden units
class RNN2(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNN2, self).__init__()
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = nn.Linear(input_size + hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, input_layer, hidden):
        combined = torch.cat((input_layer, hidden), 1)
        hidden = F.relu(self.input2hidden(combined))
        output = self.input2output(combined)
        return hidden, output

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    
# Elman RNN class that takes a padded packed minibatch
class MinibatchRNN1(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MinibatchRNN1, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x, lengths):
        hidden = self.init_hidden(x.batch_sizes[0])
        x, hidden = self.rnn(x, hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        x = x[lengths - 1, np.arange(x.shape[1]), :]
        output = self.linear(x)
        return output.flatten()

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    
class MinibatchRNN2(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MinibatchRNN2, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x, lengths):
        hidden = self.init_hidden(x.batch_sizes[0])
        x, hidden = self.rnn(x, hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        x = x[lengths - 1, np.arange(x.shape[1]), :]
        x = F.relu(x)
        output = self.linear(x)
        return output.flatten()

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    
class MinibatchLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MinibatchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x, lengths):
        hidden = self.init_hidden(x.batch_sizes[0])
        cell_state = self.init_cell(x.batch_sizes[0])
        x, hidden = self.lstm(x, (hidden, cell_state))
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        x = x[lengths - 1, np.arange(x.shape[1]), :]
        x = F.relu(x)
        output = self.linear(x)
        return output.flatten()

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
    
    def init_cell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
    