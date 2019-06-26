#!/usr/bin/env python
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class WarpedSinusoids(Dataset):
    """
    Read in Warped Sinusoids Data

    Examples
    --------
    >>> x = WarpedSinusoids()
    >>> plt.scatter(np.arange(len(x[0][1])), x[0][1])
    """
    def __init__(self, times_path=None, values_path=None):
        super(WarpedSinusoids).__init__()

        if not times_path:
            times_path = "../../data/sinusoid/times.csv"

        if not values_path:
            values_path = "../../data/sinusoid/values.csv"

        # read saved warped sinusoid data from file
        self.times = pd.read_csv(times_path).values.astype("float32")
        self.times = torch.from_numpy(self.times).unsqueeze(2)
        self.values = pd.read_csv(values_path).values.astype("float32")
        self.values = torch.from_numpy(self.values).unsqueeze(2)

        # define starting points for strided windows
        self.window_len = 10
        self.stride = 4

        self.ts_len = self.times.shape[1]
        start_pos = np.arange(0, self.ts_len, self.stride)
        self.start_pos = start_pos[start_pos < self.ts_len - self.window_len]
        self.n_ts = self.times.shape[0]


    def __getitem__(self, ix):
        # extract the window
        ts_ix = ix % self.n_ts
        start_ix = ix // self.n_ts
        window_ix = range(self.start_pos[start_ix], self.start_pos[start_ix] + 10)
        return self.times[ts_ix, window_ix, :], self.values[ts_ix, window_ix, :]


    def __len__(self):
        return self.n_ts * len(self.start_pos)
