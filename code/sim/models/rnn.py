#!/usr/bin/env python
"""
Classes of RNNs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceModel(nn.Module):
    """
    Wrapper for RNN Models

    Just a light wrapper around nn.RNN, nn.GRU, and nn.LSTM. I'm not exactly
    sure how we're going to extract gating values from these, but we'll get to
    that later.

    Examples
    --------
    >>> x = DataLoader(WarpedSinusoids())
    >>> _, yminus, yplus = next(iter(x))
    >>>
    >>> # to predict the next timepoint from the previous ones
    >>> h, h_n, y_hat = SequenceModel()(y[:, :-1, :])
    >>> plt.scatter(y[:, 1:, :].detach().numpy(), y_hat.detach().numpy())
    """
    def __init__(self, unit_type="GRU"):
        super(SequenceModel, self).__init__()
        self.n_layers = 2
        self.input_size = 1
        self.output_size = 1
        self.hidden_size = 10
        params = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.n_layers
        }

        if unit_type == "RNN":
            self.featurizer = nn.RNN(**params)
        elif unit_type == "GRU":
            self.featurizer = nn.GRU(**params)
        elif unit_type == "LSTM":
            self.featurizer = nn.LSTM(**params)
        self.regressor = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x):
        h, h_n = self.featurizer(x.transpose(1, 0))
        h, h_n = h.transpose(1, 0), h_n.transpose(1, 0)
        return h, h_n, self.regressor(h)
