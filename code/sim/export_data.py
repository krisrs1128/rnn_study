#!/usr/bin/env python
"""
Write Data for Analysis
"""
from loaders import WarpedSinusoids
import torch


def sigm_factory(W1, W2, bx, bh):
    """
    Common Operation in Gated RNNs
    """
    def sigm(x, h):
        Wx = torch.matmul(W1, x)
        Wh = torch.matmul(W2, h)
        return torch.sigmoid(Wx + bx + Wh + bh)
    return sigm


def layer_params(weights_ih, weights_hh, bias_ih, bias_hh):
    zix = range(20, 30) # I know, it's hard coded...
    params = {
        "z" :{
            "Wi": weights_ih[zix, :],
            "Wh": weights_hh[zix, :],
            "bi": bias_ih[zix],
            "bh": bias_hh[zix]
        }
    }
    return params


# looking at one of the trained models
model = torch.load("../data/models/sinusoid1561562171.pt")

# default extraction
ws = WarpedSinusoids()
with torch.no_grad():
    h, h_n, y_hat = model(ws.values[:, :-1, :])
    pd.DataFrame(y_hat.squeeze().numpy()).to_csv("../data/sinusoid/y_hat.csv", index=False)

# extract the z-gating values for some artificial input
gru = model.featurizer
params = layer_params(gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0)["z"]
z0_fun = sigm_factory(params["Wi"], params["Wh"], params["bi"], params["bh"])
z0_fun(-torch.ones(1), torch.zeros(10))
