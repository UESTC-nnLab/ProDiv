"""
This code is adapted from RNN-MIL
See details in  https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019.
"""
import torch.nn as nn
import torch
class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims

        self.fc1 = nn.Linear(1024, ndims)
        self.fc2 = nn.Linear(ndims, ndims)

        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.fc1(input)
        state = self.fc2(state)
        state = self.activation(state + input)
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)