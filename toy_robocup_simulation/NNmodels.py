"""
by: Razvan Ciuca
date: oct 14 2022

This file contains the definitions for the neural network models we use as function approximators for our Q-Learning
algorithm.

See https://www.deeplearningbook.org/ chapter 6 and https://pytorch.org/tutorials/ for documentation and tutorials

"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    This class defines an MLP with arbitrary layers and nodes
    """
    def __init__(self, nodes):
        """
        :param nodes: list containing integers corresponding to our numbers of nodes in each layer
        """
        super(MLP, self).__init__()
        self.nodes = nodes
        # initialize the weights for all layers, with std = 1/nodes[i]**0.5
        self.weights = nn.ParameterList([nn.Parameter(t.randn(nodes[i], nodes[i+1])/(nodes[i]**0.5), requires_grad=True)
                        for i in range(0, len(nodes)-1)])
        # initialize the biases to 0
        self.biases = nn.ParameterList([nn.Parameter(t.zeros(nodes[i+1]), requires_grad=True)
                       for i in range(0, len(nodes)-1)])
        # list containing our transition functions, all ReLU instead of the last one, which is just the identity
        self.sigmas = [F.relu for _ in range(0, len(self.weights)-1)] + [lambda x: x]

    def forward(self, inputs):
        """
        :param inputs: assumed to be of size [batch_size, self.nodes[0]]
        :return: returns the output tensor, of size [batch_size, self,nodes[-1]]
        """

        x = inputs

        for w, b, sigma in zip(self.weights, self.biases, self.sigmas):
            x = sigma(x @ w + b)

        return x

if __name__ == "__main__":

    # this is just a dummy test to make sure everything works fine

    batch_size = 64
    model = MLP([10, 20, 20, 10])
    inputs = t.randn(batch_size, 10)
    outputs = model(inputs)
    objective = outputs.mean()
    objective.backward()

    print(list(model.parameters())[0].std())



