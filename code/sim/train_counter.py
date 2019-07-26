#!/usr/bin/env python
"""
Training Sequence Models
"""
from loaders import CounterData
from models.counter import CounterModel
from torch.utils.data import DataLoader
import pandas as pd
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        for _, x, count in loaders[phase]:
            _, _, y_hat = model(x.to(device))
            loss = loss_fun(y_hat.squeeze(1), count.to(device))
            errors.append({
                "y": count.detach().numpy(),
                "y_hat": y_hat.squeeze(1).detach().numpy(),
                "phase": phase
            })

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), pd.concat([pd.DataFrame(s) for s in errors])


if __name__ == '__main__':
    opts = json.load(open("opts.json", "r"))
    model = CounterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=opts["train"]["lr"])
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
