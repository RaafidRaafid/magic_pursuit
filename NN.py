import torch
import torch.nn as nn
import math

from torch.nn.modules.module import Module
import torch.nn.functional as F

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class GraphConv(Module):
    def __init__(self, in_features, out_features, activation=None, bnorm=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation
        self.bnorm = bnorm
        if self.bnorm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, inp):
        x = inp[0]
        laplacian = inp[1]

        # print(x.shape)
        x = torch.matmul(laplacian, x)

        x = self.fc(x)

        if self.bnorm:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return [x, laplacian]


def generate_laplacian(A):
    A = torch.FloatTensor(A)
    A = A.to(device)
    N = A.shape[0]
    I = torch.eye(N).to(device)
    A_hat = A
    # ~~~ To increase self importance
    # A_hat = A + I
    D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
    L = D_hat * A_hat * D_hat
    return L


class BackboneNN(Module):
    def __init__(self, in_channel, out_channel, adj, filters=[128, 128], bnorm=False, n_hidden=0, noGcn=False,
                 debugging=False):
        super(BackboneNN, self).__init__()
        gconv = []
        for layer, f in enumerate(filters):
            if layer == 0:
                gconv.append(GraphConv(in_features=in_channel, out_features=f,
                                       # activation=nn.ReLU(inplace=True),
                                       activation=nn.Tanh(),
                                       bnorm=bnorm))
            else:
                gconv.append(GraphConv(in_features=filters[layer - 1], out_features=f,
                                       # activation=nn.ReLU(inplace=True),
                                       activation=nn.Tanh(),
                                       bnorm=False))
        gconv.append(GraphConv(in_features=filters[-1], out_features=out_channel,
                               activation=nn.ReLU(inplace=True),
                               # activation=nn.Tanh(),
                               bnorm=False))
        self.gconv = nn.Sequential(*gconv)

        if noGcn:
            self.laplacian = torch.eye(adj.shape[0])
        else:
            self.laplacian = generate_laplacian(adj)

    def forward(self, x):
        x = self.gconv([x, self.laplacian])[0]
        return x

    def step(self, x):
        x = torch.FloatTensor(x)
        x = x.to(device)
        return self.forward(x).to(device)


class PredictionNN(Module):
    def __init__(self, in_channel, out_channel, n_hidden=64, dropout=0.2, debugging=False):
        super(PredictionNN, self).__init__()

        self.debugging = debugging

        # ~~~ dropout experiment
        fcPolicy = []
        fcQ = []
        if n_hidden == 0:
            fcPolicy.append(nn.Linear(in_channel, out_channel))
            fcQ.append(nn.Linear(in_channel, out_channel))
        else:
            fcPolicy.append(nn.Linear(in_channel, n_hidden))
            fcPolicy.append(nn.ReLU(inplace=True))
            fcPolicy.append(nn.Linear(n_hidden, out_channel))

            fcQ.append(nn.Linear(in_channel, n_hidden))
            fcQ.append(nn.ReLU(inplace=True))
            fcQ.append(nn.Linear(n_hidden, 1))

        self.fcPolicy = nn.Sequential(*fcPolicy)
        self.fcQ = nn.Sequential(*fcQ)

    def forward(self, x, depth):

        # ~~~ add depth
        x = torch.max(x, dim=-2)[0]

        Policy = self.fcPolicy(x)
        Q = self.fcQ(x)

        softmax = nn.Softmax(dim=0)

        return Policy, softmax(Policy), Q

    def step(self, x, depth):
        # x = torch.FloatTensor(x)
        x = x.to(device)
        _, pi, Q = self.forward(x, depth)
        return pi, Q[0]
