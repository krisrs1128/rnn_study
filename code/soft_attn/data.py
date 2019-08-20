#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os.path

class AvgNeighbors(Dataset):
    """
    Data to Learn Average the Neighbors

    Examples
    --------
    >>> AvgNeighbors("../data/sinusoid/train")
    >>> plt.scatter(np.arange(len(x[0][1])), x[0][1])
    """
    def __init__(self, data_dir=None, slope=.3, M=10):
        super(AvgNeighbors).__init__()

        if not data_dir:
            data_dir = "../data/sinusoid/"

        times_path = os.path.join(data_dir, "times.csv")
        values_path = os.path.join(data_dir, "values.csv")

        # read saved warped sinusoid data from file
        self.times = pd.read_csv(times_path).values.astype("float32")
        self.times = torch.from_numpy(self.times).unsqueeze(2)
        self.values = pd.read_csv(values_path).values.astype("float32")
        self.values = torch.from_numpy(self.values).unsqueeze(2)
        self.indics = np.bitwise_and(self.values <= 1.0, self.values >= 0.0)

        self.response = []
        for i in range(len(self.times)):
            cur_values = (1 / M) * self.values[i][::M]
            for p in range(1, M):
                cur_values += (1 / M) * self.values[i][p::M]

            self.response.append(cur_values)

        self.response = torch.stack(self.response).squeeze()


    def __getitem__(self, ix):
        return self.times[ix], self.values[ix, :, :], self.response[ix, :]


    def __len__(self):
        return self.times.shape[0]
