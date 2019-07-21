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
    model.train()

    for _, y, count in iterator:
        optimizer.zero_grad()

        _, _, y_hat = model(y.to(device))
        loss = loss_fun(y_hat.squeeze(), count.to(device))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)


if __name__ == '__main__':
    opts = {"train": {"n_epochs": 3, "lr": 1e-3}}
    model = CounterModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["train"]["lr"])
    cd = CounterData()

    for epoch in range(opts["train"]["n_epochs"]):
        model, train_loss = train(model, DataLoader(cd, batch_size=20), optimizer, nn.CrossEntropyLoss())
        print("\tTrain Loss: {}".format(train_loss))

    torch.save(model, "sinusoid{}.pt".format(int(time.time())))
