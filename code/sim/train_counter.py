#!/usr/bin/env python
"""
Training Sequence Models
"""
from loaders import CounterData
from models.counter import CounterModel
from torch.utils.data import DataLoader
import time
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


def meval(model, iterator, loss_fun):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0
    model.eval()

    for _, x, count in iterator:
        optimizer.zero_grad()

        _, _, y_hat = model(x.to(device))
        loss = loss_fun(y_hat.squeeze(1), count.to(device))
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


if __name__ == '__main__':
    opts = {"train": {"n_epochs": 80, "lr": 1e-3}}
    model = CounterModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["train"]["lr"])
    cd = CounterData("/data/train/")
    vd = CounterData("/data/validation/")

    for epoch in range(opts["train"]["n_epochs"]):
        model, train_loss = train(model, DataLoader(cd, batch_size=20), optimizer, nn.CrossEntropyLoss())
        print("\tTrain Loss: {}".format(train_loss))

        with torch.no_grad():
            validation_loss = meval(model, DataLoader(vd, batch_size=20), nn.CrossEntropyLoss()) 

        print("Validation Loss: {}".format(validation_loss))

    torch.save(model, "sinusoid{}.pt".format(int(time.time())))
