#!/usr/bin/env python
"""
Training Sequence Models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, iterator, optimizer, loss_fun):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0
    model.train()

    for _, y in iterator:
        optimizer.zero_grad()
        yplus = y[:, 1:, :].to(device)
        yminus = y[:, :-1, :].to(device)

        _, _, y_hat = model(yminus)
        loss = loss_fun(y_hat.squeeze(1), yplus)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)


opts = {
    "train": {"n_epochs": 20, "lr": 1e-3}
}


model = SequenceModel()
optimizer = torch.optim.SGD(model.parameters(), lr=opts["train"]["lr"])
ws = WarpedSinusoids()

for epoch in range(opts["train"]["n_epochs"]):
    model, train_loss = train(model, DataLoader(ws), optimizer, nn.MSELoss())
    print("\tTrain Loss: {}".format(train_loss))

h_final, h, y_hat = model(ws.values[:, :-1, :])
pd.DataFrame(y_hat.squeeze().detach().numpy()).to_csv("../../data/sinusoid/y_hat.csv", index=False)
