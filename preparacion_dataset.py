#Core Imports
from VanillaGCN import datanetAPI, data_generator

#System Import
import os

#Pytorch 
import torch
from torch_geometric.data import Data

#Math and transforming imports
import numpy as np
import random
import matplotlib.pyplot as plt

#Networkx
import networkx as nx
from astropy.visualization import hist

#MatPlotLib
import matplotlib.pyplot as plt


def preparation_dataset(src_path):

    # Range of the maximum average lambda | traffic intensity used
    #  max_avg_lambda_range = [min_value,max_value]
    max_avg_lambda_range = [10,10000]
    # List of the network topology sizes to use
    net_size_lst = [9]
    # Obtain all the samples from the dataset
    reader = datanetAPI.DatanetAPI(src_path,max_avg_lambda_range, net_size_lst)
    samples_lst = []
    in_data = []
    out_data = []
    edge_index = []

    for sample in reader:
        samples_lst.append(sample)
        S = sample.get_performance_matrix()
        R = sample.get_traffic_matrix()
        
        input_to_tensor = []
        delays_lst = []
        jitter_lst = []
        pkts_gen_lst = []
        for i in range (sample.get_network_size()):
            cumulativeDelay = 0
            cumulativeJitter = 0
            cumulativePkts_gen = 0
            for j in range (sample.get_network_size()):
                if (i == j):
                    continue
                cumulativeDelay = S[i,j]["AggInfo"]["AvgDelay"] + cumulativeDelay
                cumulativeJitter = S[i,j]["AggInfo"]["Jitter"] + cumulativeJitter

            for j in range (sample.get_network_size()):
                if (i == j):
                    continue
                cumulativePkts_gen = R[i,j]["AggInfo"]["TotalPktsGen"] + cumulativePkts_gen
            #Node i
            delays_lst.append(cumulativeDelay)
            jitter_lst.append(cumulativeJitter)
            pkts_gen_lst.append(cumulativePkts_gen)

        #InputData
        aux_in_lst = [delays_lst,jitter_lst, pkts_gen_lst]
        metricas_in = np.asarray(aux_in_lst)
        input_to_tensor = torch.Tensor(metricas_in)
        in_data.append(input_to_tensor)

        #Adjacency Matrix
        G = nx.DiGraph(sample.get_topology_object())
        edge_index.append(nx.adjacency_matrix(G))

    in_data_tensor = torch.stack(in_data)

    return in_data_tensor, edge_index
    
def hg_to_data (HG):
    dic_HG = []
    dic_y_t = []
    adjacency = []
    for i in range(len(HG)):
        dic_HG.append({"capacity": np.expand_dims(list(nx.get_node_attributes(HG[i], 'capacity').values()), axis=1),
            "queue_size": np.expand_dims(list(nx.get_node_attributes(HG[i], 'queue_size').values()), axis=1)
            })
    for i in range(len(HG)):
        dic_y_t.append(np.expand_dims(list(nx.get_node_attributes(HG[i], 'delay').values()), axis = 1))
    
    for i in range(len(HG)):
        adjacency.append(nx.to_numpy_matrix(HG[i]))

    return dic_HG, dic_y_t, adjacency

def prepare_data(input, edge_index, labels):
    """
    Prepare data for the GCN model
    Args:
        input: Input metrics for the model (Tensor)
        edge_index: All Adjacency Matrices in the requiered form for GCNConv (Tensor)
        labels: Output data to compare (Tensor)
    Returns:
        data: The data ready to be fed to the GCN model
    """
    #For the case of a GCNConv, we will only use one topology
    a = edge_index[0].todense()
    edge_tensor = torch.tensor(a, dtype = torch.long)
    input_edge_tensor = edge_tensor.nonzero().t().contiguous()
    
    data = Data(x=input, edge_index=input_edge_tensor, y=labels)

    return data

def plot_mse_epoch(epoch, mse_loss):
    """
    Plot MSExEpoch
    Args:
        epoch: number of epoch
        mse_loss: The MSE values to be plotted
    """
    fig, ax = plt.subplots()

    plt.title("MSE per Iteration")
    plt.legend()
    plt.plot(range(epoch),mse_loss, color='red', marker='.')
    
    plt.savefig('Epoch.png')