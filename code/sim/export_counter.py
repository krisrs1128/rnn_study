#!/usr/bin/env python
from sklearn.manifold import MDS
from loaders import CounterData
from export_data import layer_params
import numpy as np
import torch
import json

def rounded_list(x, prec=4):
    x = x.tolist()
    for i in range(len(x)):
        if type(x[i]) == list:
            for j in range(len(x[i])):
                x[i][j] = round(x[i][j], prec)
        else:
            x[i] = round(x[i], prec)
    return x


def reshape_x(x):
    result = []
    for i in range(len(x)):
        if i % 5 == 0:
            result.append({
                "time": i,
                "value": round(x[i].item(), 4)
            })

    return result

if __name__ == '__main__':
    model = torch.load("sinusoid1563895472.pt", map_location="cpu")
    D = model.featurizer.state_dict()
    params_dict = {}

    # Extract the GRU data
    n_layers = 4
    for l in range(n_layers):
        params_dict["l" + str(l)] = layer_params(D["weight_ih_l" + str(l)], D["weight_hh_l" + str(l)], D["bias_ih_l" + str(l)], D["bias_hh_l" + str(l)])

    for k1, v1 in params_dict.items():
        for k2, v2 in v1.items():
            for k3, v3 in v2.items():
                params_dict[k1][k2][k3] = rounded_list(params_dict[k1][k2][k3])

    json.dump(params_dict, open("gru.json", "w"))


    # Compute the MDS data
    data = CounterData("../../data/sinusoid/")
    dim_h = 10
    n = 1000
    hmat = np.zeros((n, dim_h))
    ix = np.random.randint(0, len(data), n)
    mds = MDS()

    with torch.no_grad():
        for i in range(n):
            if i % 10 == 0: print(i)
            h, h_n, y_hat = model(data[ix[i]][0].unsqueeze(0))
            hmat[i, :] = h_n[0, -1, :]

        mds.fit(np.arctanh(hmat))
        embedding = []

        for i in range(n):
            embedding.append({
                "mds": rounded_list(mds.embedding_[i, :]),
                "x": reshape_x(data[i][1].numpy()),
                "counts": data[i][2].item()
            })

        json.dump(embedding, open("mds.json", "w"))
