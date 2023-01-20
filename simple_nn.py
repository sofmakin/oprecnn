import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# A simple neural network model with linear layers and ReLU-layer as an activation function
# Model for structure adapted and modified from pytroch tutorials

class NeuralNetwork(nn.Module):
    def __init__(self, dim, num_nodes):
        super(NeuralNetwork, self).__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        
        # define the layers used
        self.linear1 = nn.Linear(dim, num_nodes)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(num_nodes, 2)

        
    def forward(self, x): # in the forward step define how the data passes through
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
     
        return x