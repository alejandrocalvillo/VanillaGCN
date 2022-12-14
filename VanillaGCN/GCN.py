import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
#Para ver que tal lo hemos hecho loss-epoch
#from tensorboardX import SummaryWriter 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class VanillaGCN(nn.Module):
    #input_dim: Nº of input dimensions
    #output_dim: Nº of output dimensions
    #num_layers: Nº of convolution performed
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.conv1 = GCNConv(input_size, 16)
        self.conv2 = GCNConv(16, hidden_size)


#     == Abajo el código de la documentación ==
#     input para cada nodo (x), matriz de adyacencia (Adj)
#     y pesos de los enlaces (edge_index)
#
#     def forward(self, x: Tensor, edge_index: Adj,
#                 edge_weight: OptTensor = None) -> Tensor:

    def forward(self, data):
        x, edge_index = data['traffic'], data['path-to-queue']

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    # def __init__(self, input_dim, output_dim, output_dim_data, num_layers=2) -> None:
    #     super(VanillaGCN,self).__init__
    #     self.conv=nn.ModuleList()
    #     self.conv.append(pyg_nn.GCNConv(input_dim, output_dim))
    #     for i in range(num_layers):
    #         self.conv.append(pyg_nn.GCNConv(output_dim, output_dim))

    #     #We will use this a a sequential linear layers model if needed
    #     self.dropout = 0.25    
    #     self.post_mp = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Dropout(self.dropout), nn.Linear(output_dim, output_dim_data))


    # def forward(self, data):
    #     # data.x: feature matrix
    #     # data.edge_index: adjacency matrix
    #     # data.batch: batch matrix (A matrix that represent relation between nodes belonging to a graph)
    #     x, edge_index, batch = data.x, data.edge_index, data.batch

    #     for i in range(self.num_layers):
    #         x=self.conv[i](x, edge_index)
    #         embending=x
    #         #Probar con reLU tambien
    #         x=F.tanh(x)
    #         x = self.dropout(x, p=self.dropout, training=self.training)
    #         if not i == self.num_layers - 1:
    #             x = self.lns[i](x)
            
    #         x = self.post_mp(x)

    #     return embending, F.log_softmax(x, dim=1)
    
    # def loss(self, pred, true):
    #     return F.nll_loss(pred, true)
