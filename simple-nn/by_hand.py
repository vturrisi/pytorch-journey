# start hour 14:56 (22 jan)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
np.random.seed(1)


class MLP:
    def __init__(self, arch):
        self.W = [torch.randn(inp, out, requires_grad=True)
                  for inp, out in zip(arch[:-1], arch[1:])]
        self.biases = [torch.zeros(i, requires_grad=True, dtype=torch.float) for i in arch[1:]]

    def forward(self, X):
        o = X
        for W, bias in zip(self.W, self.biases):
            o = F.relu(torch.mm(o, W) + bias)
        return F.softmax(o, dim=1)  # applies softmax

    def forward_without_softmax(self, X):
        o = X
        for W, bias in zip(self.W, self.biases):
            o = F.relu(torch.mm(o, W) + bias)
        return o

    def backward(self, yhat, y):
        self.manual_zero_grad()

        y_ = torch.tensor([1, 0, 0], requires_grad=True, dtype=torch.float)
        loss = - torch.sum(y_ * torch.log(yhat.view(-1)))
        print(loss)
        loss.backward(retain_graph=True)
        print(self.W[0].grad)

    def auto_backward(self, yhat, y):
        self.manual_zero_grad()

        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(yhat, y)
        print(loss)
        loss.backward(retain_graph=True)
        print(self.W[0].grad)

    def manual_zero_grad(self):
        for p in self.W:
            if p.grad is not None:
                p.grad.data.zero_()

        for p in self.biases:
            if p.grad is not None:
                p.grad.data.zero_()

    def update_grad(self, grad):
        pass


if __name__ == '__main__':
    # prepare csv data
    data = pd.read_csv('../datasets/iris.csv')
    class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    for y, name in class_names.items():
        data.loc[data['species'] == name, 'species'] = y

    index = list(range(data.shape[0]))
    np.random.shuffle(index)
    data = data.iloc[index, :]

    mlp = MLP([data.shape[1] - 1, 3, 5, len(class_names)])

    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).long()
    for x, y in zip(X, Y):
        x = x.view(1, -1)
        y = y.view(-1)
        print('manual')
        yhat = mlp.forward(x)
        print(yhat)
        grad = mlp.backward(yhat, y)

        print('auto')
        yhat = mlp.forward_without_softmax(x)
        print(yhat)
        grad = mlp.auto_backward(yhat, y)
        exit()
        mlp.update_grad(grad)
    print(mlp.W)
