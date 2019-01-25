# start hour 14:56 end hour 16:20 (22 jan)
# start hour 13:20 - 13:35 | 15:10 (23 jan)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

torch.manual_seed(1)
np.random.seed(1)


class MLP:
    def __init__(self, arch, learning_rate=0.01):
        self.W = [torch.rand(inp, out, requires_grad=True)
                  for inp, out in zip(arch[:-1], arch[1:])]
        self.biases = [torch.zeros(i, requires_grad=True, dtype=torch.float) for i in arch[1:]]
        self.parameters = list(chain(self.W, self.biases))

        self.learning_rate = learning_rate

    def forward(self, X):
        o = X
        for W, bias in zip(self.W[:-1], self.biases[:-1]):
            o = F.relu(torch.mm(o, W) + bias)
        o = torch.mm(o, self.W[-1]) + self.biases[-1]
        # needs to use log_softmax due to numerical instability
        o = F.log_softmax(o, dim=1)
        return o

    def forward_without_softmax(self, X):
        o = X
        for W, bias in zip(self.W[:-1], self.biases[:-1]):
            o = F.relu(torch.mm(o, W) + bias)
        o = torch.mm(o, self.W[-1]) + self.biases[-1]
        return o

    def backward(self, yhat, y):
        y = y.item()
        if y == 0:
            y_ = torch.tensor([1, 0, 0], requires_grad=True, dtype=torch.float)
        elif y == 1:
            y_ = torch.tensor([0, 1, 0], requires_grad=True, dtype=torch.float)
        elif y == 2:
            y_ = torch.tensor([0, 0, 1], requires_grad=True, dtype=torch.float)

        loss = - torch.sum(y_ * yhat.view(-1))
        loss.backward()
        return [p.grad for p in self.parameters]

    def auto_backward(self, yhat, y):
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(yhat, y)
        loss.backward()
        return [p.grad for p in self.parameters]

    def manual_zero_grad(self):
        for p in self.W:
            if p.grad is not None:
                p.grad.data.zero_()

        for p in self.biases:
            if p.grad is not None:
                p.grad.data.zero_()

    def update_grad(self, grads):
        for p, grad in zip(self.parameters, grads):
            p.data.sub_(self.learning_rate * grad)

    def auto_update_grad(self):
        for p in self.parameters:
            p.data.sub_(self.learning_rate * p.grad)


if __name__ == '__main__':
    # prepare csv data
    data = pd.read_csv('../datasets/iris.csv')
    class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    for y, name in class_names.items():
        data.loc[data['species'] == name, 'species'] = y

    mlp = MLP([data.shape[1] - 1, 10, 5, len(class_names)])
    # mlp_auto = MLP([data.shape[1] - 1, 30, 5, len(class_names)])

    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    # convert to tensor
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()

    for epoch in range(10):
        # shuffle data
        index = torch.randperm(X.size(0))
        X = X[index]
        Y = Y[index]

        rights = 0
        total = X.size(0)
        for i, (x, y) in enumerate(zip(X, Y)):
            mlp.manual_zero_grad()
            # mlp_auto.manual_zero_grad()

            x = x.view(1, -1)
            y = y.view(-1)

            yhat = mlp.forward(x)
            grads = mlp.backward(yhat, y)
            mlp.update_grad(grads)

            # yhat = mlp_auto.forward_without_softmax(x)
            # auto_grads = mlp_auto.auto_backward(yhat, y)
            # mlp_auto.auto_update_grad()

            # assert [g1 == g2 for g1, g2 in zip(grads, auto_grads)]
            # assert [p1 == p2 for p1, p2 in zip(mlp.parameters, mlp_auto.parameters)]

            v, prediction = torch.max(yhat, 1)
            prediction = prediction.item()
            if prediction == y.item():
                rights += 1

        print(rights / total)
