import math
import sys
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')
from utils.load_csv_dataset import load_iris

torch.manual_seed(1)
np.random.seed(1)


class MLP:
    def __init__(self, arch, learning_rate=0.01):
        # perform Xavier initialisation random from [0, 1) and divide by sqrt(input size)
        # https://pytorch.org/docs/stable/torch.html#torch.rand
        self.W = [torch.rand(inp, out) / math.sqrt(inp)
                  for inp, out in zip(arch[:-1], arch[1:])]
        # turn grads on after Xavier initialisation (to not keep track of the sqrt operation)
        [W.requires_grad_() for W in self.W]
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
        # o = self.log_softmax(o)  # manual log softmax
        return o

    @staticmethod
    def log_softmax(x):
        # ln(e^x / (e^x + e^x+1 + e^x+2)) ->
        #   ln(e^x) - ln(e^x + e^x+1 + e^x+2) ->
        #   x - ln(e^x + e^x+1 + e^x+2)
        return x - x.exp().sum(-1).log().unsqueeze(-1)

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

        loss = - torch.sum(y_ * yhat.view(-1))  # manual negative likelyhood
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
    X, Y, class_names = load_iris()

    mlp = MLP([X.size(1), 10, 5, len(class_names)])
    # mlp_auto = MLP([data.shape[1] - 1, 30, 5, len(class_names)])

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
