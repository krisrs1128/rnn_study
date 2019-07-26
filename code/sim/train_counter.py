#!/usr/bin/env python
"""
Training Sequence Models
"""
from loaders import CounterData
from models.counter import CounterModel
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn


def train(model, iterator, optimizer, loss_fun):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0
    model = model.to(device)
    model.train()

    for _, x, count in iterator:
        optimizer.zero_grad()

        _, _, y_hat = model(x.to(device))
        loss = loss_fun(y_hat.squeeze(), count.to(device))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)


def meval(model, loaders, loss_fun):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    errors = []
    epoch_loss = 0
    model.eval()

    for phase in ["train", "validation"]:
        i = 0
        for _, x, count in loaders[phase]:
            _, _, y_hat = model(x.to(device))
            errors.append({
                "index": np.arange(len(count)),
                "y": count.detach().numpy(),
                "y_hat": y_hat.squeeze(1).detach().cpu().numpy(),
                "phase": phase
            })
            i += 1

            if phase == "validation":
                loss = loss_fun(y_hat.squeeze(1), count.to(device))
                epoch_loss += loss.item()

    return epoch_loss / len(loaders["validation"]), pd.concat([pd.DataFrame(s) for s in errors])


if __name__ == '__main__':
    opts = json.load(open("/home/code/sim/opts.json", "r"))
    model = CounterModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["train"]["lr"])
    loaders = {
        "train": DataLoader(CounterData(opts["data"]["train"]), batch_size=32),
        "validation": DataLoader(CounterData(opts["data"]["validation"]), batch_size=32)
    }

    for epoch in range(opts["train"]["n_epochs"]):
        model, train_loss = train(model, loaders["train"], optimizer, nn.MSELoss())

        with torch.no_grad():
            valid_loss, errors = meval(model, loaders, nn.MSELoss())
            errors["epoch"] = epoch
            errors.to_csv("progress.csv", index=False, mode="a", header=False)

        print("Epoch: {}\tTrain: {}\tValidation: {}".format(epoch, train_loss, valid_loss))

    torch.save(model, "sinusoid{}.pt".format(int(time.time())))
