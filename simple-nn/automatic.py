# start hour 14:56 end hour 16:20 (22 jan)
# start hour 13:20 - 13:35 | 15:10 (23 jan)

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

torch.manual_seed(1)
np.random.seed(1)


class MLP(nn.Module):
    def __init__(self, arch):
        super().__init__()

        self.layers = nn.Sequential()
        for i, (inp, out) in enumerate(zip(arch[:-2], arch[1:-1])):
            self.layers.add_module('linear-{}'.format(i + 1), nn.Linear(inp, out))
            self.layers.add_module('relus-{}'.format(i + 1), nn.ReLU())
        self.layers.add_module('linear-{}'.format(i + 2), nn.Linear(arch[-2], arch[-1]))

    def forward(self, X):
        o = self.layers(X)
        return o
        # return F.softmax(self.layers(X), dim=1)

if __name__ == '__main__':
    # prepare csv data
    data = pd.read_csv('../datasets/iris.csv')
    class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    for y, name in class_names.items():
        data.loc[data['species'] == name, 'species'] = y

    loss_function = nn.CrossEntropyLoss()
    mlp_auto = MLP([data.shape[1] - 1, 20, 50, 20, 1000, 100, len(class_names)])
    optimiser = optim.SGD(mlp_auto.parameters(), lr=0.001, momentum=0.9)

    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    # convert to tensor
    X = torch.from_numpy(X).float()
    std = torch.std(X, 0)
    mean = torch.mean(X, 0)

    X = (X[:] - mean) / std
    Y = torch.from_numpy(Y).long()

    mlp_auto.train()
    for epoch in range(10):
        # shuffle data
        index = torch.randperm(X.size(0))
        X = X[index]
        Y = Y[index]

        rights = 0
        total = X.size(0)
        with torch.set_grad_enabled(True):
            for i, (x, y) in enumerate(zip(X, Y)):

                x = x.view(1, -1)
                y = y.view(-1)

                yhat = mlp_auto(x)
                loss = loss_function(yhat, y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                # print(mlp_auto.layers[-1].weight)

                v, prediction = torch.max(yhat, 1)
                prediction = prediction.item()
                if prediction == y.item():
                    rights += 1

        print(rights / total)
