#!/usr/bin/env python
"""
Classes of RNNs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CounterModel(nn.Module):
    """
    Wrapper for RNN Models

    Just a light wrapper around nn.RNN, nn.GRU, and nn.LSTM. I'm not exactly
    sure how we're going to extract gating values from these, but we'll get to
    that later.

    Examples
    --------
    >>> x = DataLoader(CounterData())
    >>> _, y, count = next(iter(x))
    >>>
    >>> # to predict the next timepoint from the previous ones
    >>> h, h_n, y_hat = CounterModel()(y[:, :, :])
    >>> plt.scatter(y[:, 1:, :].detach().numpy(), y_hat.detach().numpy())
    """
    def __init__(self):
        super(CounterModel, self).__init__()
        self.n_layers = 5
        self.input_size = 1
        self.output_size = 1
        self.hidden_size = 20
        params = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.n_layers
        }

        self.featurizer = nn.GRU(**params)
        self.regressor = nn.Linear(self.n_layers * self.hidden_size, self.output_size)


    def forward(self, x):
        h, h_n = self.featurizer(x.transpose(1, 0))
        h, h_n = h.transpose(1, 0), h_n.transpose(1, 0)
        return h, h_n, self.regressor(h_n.flatten(1))
