#!/usr/bin/env python
"""
Write Data for Analysis
"""
from loaders import WarpedSinusoids
import pandas as pd
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
    """
    Evaluate Units Across Sequence

    This calls cell_stack along a sequence of inputs. It makes all the GRU
    activation values (h, n, r, z) visible, at every layer, and every
    timepoint.

    :param: f_stack: A list of GRU cell functions, corresponding to the learned
      cells at every layer for any particular timepoint.
    :param x_seq: An input x sequence. Assumed to have dimension timepoint x feature_dim.
    :param h_prev: A list giving the initial hidden unit values at every layer.
      Should have the same length as f_stack.
    :return A nested collection of dictionaries. The outer dictionary is
      indexed by time (t0, t1, ...), the next inner one indexes layers (l0, the
      lowest / closest to input, to lk, the highest), and the inner one gives
      the different types of activations, z and r for gatings, and n and h for
      (pre)updated hidden units.

    Examples
    --------
    >>> # looking at one of the trained models
    >>> model = torch.load("../../data/models/sinusoid1561562171.pt")
    >>> gru = model.featurizer
    >>> h, hn = gru(torch.zeros((5, 1, 1)))
    >>>
    >>> # computations "by hand" agree
    >>> params = []
    >>> params.append(layer_params(gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0))
    >>> params.append(layer_params(gru.weight_ih_l1, gru.weight_hh_l1, gru.bias_ih_l1, gru.bias_hh_l1))
    >>> stack_fun = cell_stack(params)
    >>> h0 = [torch.zeros((10, 1)), torch.zeros((10, 1))]
    >>> outputs = cell_seq(stack_fun, torch.zeros((5, 1)), h0)

    """
    time_len = x_seq.shape[0]
    outputs = {}

    for i in range(time_len):
        ix = "t{}".format(i)
        outputs[ix] = f_stack(x_seq[i:(i + 1), :], h_prev)
        K = len(outputs[ix])
        h_prev = [outputs[ix]["l{}".format(k)]["h"] for k in range(K)]

    return outputs


if __name__ == '__main__':
    # load data and model
    ws = WarpedSinusoids("../data/sinusoid/")
    model = torch.load("../data/models/sinusoid1561562171.pt")

    # get parameters that define the GRU cells
    gru = model.featurizer
    params = []
    params.append(layer_params(gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0))
    params.append(layer_params(gru.weight_ih_l1, gru.weight_hh_l1, gru.bias_ih_l1, gru.bias_hh_l1))
    stack_fun = cell_stack(params)
    K = params[0]["r"]["Wi"].shape[0]
    h0 = [torch.zeros((10, 1)), torch.zeros((10, 1))]

    # extract and write activations, one sample at a time
    for i in range(len(ws)):
        x = ws[i][1].unsqueeze(1)
        outputs = cell_seq(stack_fun, x, h0)

        ix = str(i).zfill(3)
        outfile = open("activations_{}.csv".format(ix), "w")

        for time, stack in outputs.items():
            for layer, cells in stack.items():
                for parameter, value in cells.items():
                    for k, v in enumerate(value.squeeze()):
                        outfile.writelines("{}\t{}\t{}\t{}\t{}\n".format(time, layer, parameter, k, v))

        outfile.close()
        print("{}/{}".format(i, len(ws)))

        # also write the predictions
        _, _, y_hat = model(x.transpose(1, 0))
        pd.DataFrame(y_hat.squeeze().detach().numpy()).to_csv("y_hat_{}.csv".format(ix))
