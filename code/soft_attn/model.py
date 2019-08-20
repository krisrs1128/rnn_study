#!/usr/bin/env python
from data import AvgNeighbors
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.featurizer = nn.GRU(input_size, hidden_size)

    def forward(self, x):
        output, hidden = self.featurizer(x.transpose(1, 0))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, seq_len=200):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len= seq_len

        self.attn_w1 = nn.Linear(self.input_size, self.seq_len)
        self.attn_w2 = nn.Linear(self.hidden_size, self.seq_len)
        self.attn_combine = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.featurizer = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, y, hidden, encoder_outputs):
        attn_weights = F.softmax(self.attn_w1(y) + self.attn_w2(hidden.squeeze(1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs) # sum alpha_i * h_i

        features = torch.cat((y, attn_applied.squeeze(1)), 1)
        features = F.relu(self.attn_combine(features))
        features, hidden = self.featurizer(features.unsqueeze(0), hidden)
        return self.out(features), hidden, attn_weights


def train_sample(x, y, models, optimizers, loss_fun, target_len=20):
    for k in optimizers.keys():
        optimizers[k].zero_grad()

    h, h_n = models["encoder"](x)

    loss = 0
    dh = torch.zeros_like(h_n)
    for i in range(target_len):
        cur_y = torch.zeros(1, 1) if i == 0 else y[:, i - 1].unsqueeze(0)
        dpred, dh, attn = models["decoder"](cur_y, dh, h.transpose(0, 1))
        loss += loss_fun(dpred.squeeze(), y[:, i].squeeze())

    loss.backward()
    for k in optimizers.keys():
        optimizers[k].step()

    return loss.item() / target_len


def train(models, data_loader, opts):
    optimizers = {
        "encoder": optim.SGD(models["encoder"].parameters(), lr=opts["lr"]),
        "decoder": optim.SGD(models["decoder"].parameters(), lr=opts["lr"]),
    }

    loss_fun = nn.MSELoss()
    for epoch in range(opts["n_epochs"]):
        loss = 0
        i = 0
        for _, x, y in data_loader:
            print(i)
            i += 1
            loss += train_sample(x, y, models, optimizers, loss_fun)

        print(f"epoch: {epoch}\t loss: {loss / len(data_loader)}")

    return models, loss

models = {"encoder": EncoderRNN(1, 32), "decoder": AttnDecoderRNN(1, 32)}
opts = {"lr": 0.001, "n_epochs": 500}
dataset = AvgNeighbors("../../data/sinusoid/train")
loader = DataLoader(dataset, sampler=SubsetRandomSampler(list(range(100))))
models, loss = train(models, loader, opts)

torch.save(models["encoder"].state_dict(), f="attn_encoder.pt")
torch.save(models["decoder"].state_dict(), f="attn_decoder.pt")
