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
        h = (1 - z) * n + z * h
        return h, n, z, r
    return f


def layer_params(weights_ih, weights_hh, bias_ih, bias_hh):
    """
    Rearrange Weight Output from Torch
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


def cell_stack(params_list):
    """
    Inspect Stack of GRU Cells

    Extracts values of h, n, z, and r, for all layers at a single timepoint.
    """
    cell_funs = []
    for i in range(len(params_list)):
        funs = gru_funs(params_list[i])
        cell_funs.append(gru_cell(*funs))

    def f(x, h_prev):
        h_in = x
        outputs = {}
        for i in range(len(cell_funs)):
            with torch.no_grad():
                h, n, z, r = cell_funs[i](h_in, h_prev[i])
                outputs["l{}".format(i)] = {"h": h, "n": n, "z": z, "r": r}
                h_in = h

        return outputs
    return f


def cell_seq(f_stack, x_seq, h_prev):
    time_len = x_seq.shape[0]
    outputs = {}

    for i in range(time_len):
        ix = "t{}".format(i)
        outputs[ix] = f_stack(x_seq[i:(i + 1), :], h_prev)
        K = len(outputs[ix])
        h_prev = [outputs[ix]["l{}".format(k)]["h"] for k in range(K)]

    return outputs


# looking at one of the trained models
model = torch.load("../../data/models/sinusoid1561562171.pt")

# default extraction
ws = WarpedSinusoids("../../data/sinusoid/")
with torch.no_grad():
    h, h_n, y_hat = model(ws.values[:, :-1, :])
    pd.DataFrame(y_hat.squeeze().numpy()).to_csv("../../data/sinusoid/y_hat.csv", index=False)

# verify that these functions agree
gru = model.featurizer
h, hn = gru(torch.zeros((5, 1, 1)))
h[0]

# computations "by hand" agree
params = []
params.append(layer_params(gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0))
params.append(layer_params(gru.weight_ih_l1, gru.weight_hh_l1, gru.bias_ih_l1, gru.bias_hh_l1))
stack_fun = cell_stack(params)
h0 = [torch.zeros((10, 1)), torch.zeros((10, 1))]
outputs = cell_seq(stack_fun, torch.zeros((5, 1)), h0)
