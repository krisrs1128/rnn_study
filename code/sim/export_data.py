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
    bx = bx.unsqueeze(1) # 1D -> column vector
    bh = bh.unsqueeze(1)

    def sigm(x, h):
        Wx = torch.matmul(W1, x)
        Wh = torch.matmul(W2, h)
        return torch.sigmoid(Wx + bx + Wh + bh)
    return sigm


def n_factory(W1, W2, bx, bh):
    """
    Generate n(x, h, r) function

    Computation for pre-gated h is different,
      - Uses tanh
      - Requires gating from r
    """
    bx = bx.unsqueeze(1)
    bh = bh.unsqueeze(1)

    def n(x, h, r):
        Wx = torch.matmul(W1, x)
        Wh = torch.matmul(W2, h)
        return torch.tanh(Wx + bx + r * (Wh + bh))
    return n


def gru_funs(params):
    """
    GRU functions from Torch Parameters
    """
    r_fun = sigm_factory(params["r"]["Wi"], params["r"]["Wh"], params["r"]["bi"], params["r"]["bh"])
    z_fun = sigm_factory(params["z"]["Wi"], params["z"]["Wh"], params["z"]["bi"], params["z"]["bh"])
    n_fun = n_factory(params["n"]["Wi"], params["n"]["Wh"], params["n"]["bi"], params["n"]["bh"])
    return r_fun, z_fun, n_fun


def gru_cell(r_fun, z_fun, n_fun):
    """
    Factory of GRU Cells
    """
    def f(x, h):
        r = r_fun(x, h)
        z = z_fun(x, h)
        n = n_fun(x, h, r)
        h = (1 - z) * n + z * zero_k
        return h, n, z, r
    return f


def layer_params(weights_ih, weights_hh, bias_ih, bias_hh):
    """
    More Readable GRU Parameters
    """
    K = weights_ih.shape[0]
    rix = range(0, K // 3)
    zix = range(K // 3, 2 * K // 3)
    nix = range(2 * K // 3, K)
    params = {
        "r" :{
            "Wi": weights_ih[rix, :],
            "Wh": weights_hh[rix, :],
            "bi": bias_ih[rix],
            "bh": bias_hh[rix]
        },
        "z" :{
            "Wi": weights_ih[zix, :],
            "Wh": weights_hh[zix, :],
            "bi": bias_ih[zix],
            "bh": bias_hh[zix]
        },
        "n" :{
            "Wi": weights_ih[nix, :],
            "Wh": weights_hh[nix, :],
            "bi": bias_ih[nix],
            "bh": bias_hh[nix]
        }
    }
    return params


# looking at one of the trained models
model = torch.load("../../data/models/sinusoid1561562171.pt")

# default extraction
ws = WarpedSinusoids("../../data/sinusoid/")
with torch.no_grad():
    h, h_n, y_hat = model(ws.values[:, :-1, :])
    pd.DataFrame(y_hat.squeeze().numpy()).to_csv("../../data/sinusoid/y_hat.csv", index=False)

# verify that these functions agree
gru = model.featurizer
h, hn = gru(torch.zeros((50, 1, 1)))
h[0]

# computations "by hand" agree
params = layer_params(gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0)
f0 = gru_cell(*gru_funs(params))
h01, n01, z01, r01 = f0(torch.zeros((1, 1)), torch.zeros((10, 1)))
params = layer_params(gru.weight_ih_l1, gru.weight_hh_l1, gru.bias_ih_l1, gru.bias_hh_l1)
f1 = gru_cell(*(gru_funs(params)))
h11, n11, z11, r11 = f1(h01, torch.zeros((10, 1)))
