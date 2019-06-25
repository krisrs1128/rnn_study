#!/usr/bin/env python
"""
Classes of RNNs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    """



    Examples
    --------
    >>> rnn = RNN()
    >>> x = DataLoader(WarpedSinusoids())
    >>> _, y = next(iter(x))
    >>>
    >>> # to predict the next timepoint from the previous ones
    >>> _, h, y_hat = rnn(y[:, :-1, :])
    >>> plt.scatter(y[:, 1:, :].detach().numpy(), y_hat.detach().numpy())
    """
    def __init__(self):
        super(RNN, self).__init__()
        self.n_layers = 2
        self.input_size = 1
        self.output_size = 1
        self.hidden_size = 10
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers
        )
        self.regressor = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h_final, h = self.gru(x)
        return h_final, h, self.regressor(h_final)
