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
    def __init__(self, arch, learning_rate=0.01):
        super().__init__()

        self.layers = nn.Sequential()
        for i, (inp, out) in enumerate(zip(arch[:-1], arch[1:])):
            self.layers.add_module('linear-{}'.format(i + 1), nn.Linear(inp, out))
            self.layers.add_module('relus-{}'.format(i + 1), nn.ReLU())

        self.learning_rate = learning_rate

    def forward(self, X):
        o = self.layers(X)
        return F.softmax(self.layers(X), dim=1)

if __name__ == '__main__':
    # prepare csv data
    data = pd.read_csv('../datasets/iris.csv')
    class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    for y, name in class_names.items():
        data.loc[data['species'] == name, 'species'] = y

    loss = nn.NLLLoss()
    mlp_auto = MLP([data.shape[1] - 1, 2, 2, len(class_names)])
    optimiser = optim.SGD(mlp_auto.parameters(), lr=0.01, momentum=0.9)

    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    # convert to tensor
    X = torch.from_numpy(X).float()
    min_values, _ = torch.min(X, 0)
    max_values, _ = torch.max(X, 0)

    X = (X[:] - min_values) / (max_values - min_values)
    Y = torch.from_numpy(Y).long()
    mlp_auto.train()
    for epoch in range(10):
        print(mlp_auto.layers[0].weight)
        # shuffle data
        index = torch.randperm(X.size(0))
        X = X[index]
        Y = Y[index]

        rights = 0
        total = X.size(0)
        for i, (x, y) in enumerate(zip(X, Y)):
            mlp_auto.zero_grad()

            x = x.view(1, -1)
            y = y.view(-1)

            yhat = mlp_auto(x)
            l = loss(yhat, y)
            l.backward()
            print(mlp_auto.layers[0].weight.grad)
            print(mlp_auto.layers[2].weight.grad)
            print(mlp_auto.layers[4].weight.grad)
            exit()
            optimiser.step()

            v, prediction = torch.max(yhat, 1)
            prediction = prediction.item()
            if prediction == y.item():
                rights += 1

        print(rights / total)
