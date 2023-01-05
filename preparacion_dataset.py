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
        aux_in_lst = [delays_lst,jitter_lst]
        metricas_in = np.asarray(aux_in_lst)
        input_to_tensor = torch.Tensor(metricas_in)
        in_data.append(input_to_tensor)

        #OutputData

        aux_out_lst = [pkts_gen_lst]
        metricas_out = np.asarray(aux_out_lst)
        output_to_tensor = torch.Tensor(metricas_out)
        out_data.append(output_to_tensor)

        #Adjacency Matrix
        G = nx.DiGraph(sample.get_topology_object())
        edge_index.append(nx.adjacency_matrix(G))

    in_data_tensor = torch.stack(in_data)
    out_data_tensor = torch.stack(out_data)

    return in_data_tensor, out_data_tensor, edge_index
    
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

def data_creator(metricas_entrada, metricas_salida, edge_index):

    #Transforming Adjacency Matrix(edge_index) into a shape that can be interpreted for the convolution
    
    adjacency_lst = []
    data = Data(edge_index = edge_index)

    return data
