#!/usr/bin/env python
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os.path


class WarpedSinusoids(Dataset):
    """
    Read in Warped Sinusoids Data

    Examples
    --------
    >>> x = WarpedSinusoids()
    >>> plt.scatter(np.arange(len(x[0][1])), x[0][1])
    """
    def __init__(self, data_dir=None):
        super(WarpedSinusoids).__init__()

        if not data_dir:
            data_dir = "../data/sinusoid/"

        times_path = os.path.join(data_dir, "times.csv")
        values_path = os.path.join(data_dir, "values.csv")

        # read saved warped sinusoid data from file
        self.times = pd.read_csv(times_path).values.astype("float32")
        self.times = torch.from_numpy(self.times).unsqueeze(2)
        self.values = pd.read_csv(values_path).values.astype("float32")
        self.values = torch.from_numpy(self.values).unsqueeze(2)


    def __getitem__(self, ix):
        return self.times[ix], self.values[ix, :-1, :], self.values[ix, 1:, :]


    def __len__(self):
        return self.times.shape[0]
