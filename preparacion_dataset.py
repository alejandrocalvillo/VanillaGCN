from VanillaGCN import datanetAPI, data_generator
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
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
    edge_index = []

    for sample in reader:
        samples_lst.append(sample)
        S = sample.get_performance_matrix()
        input_to_tensor = []
        delays_lst = []
        jitter_lst = []

        for i in range (sample.get_network_size()):
            cumulativeDelay = 0
            cumulativeJitter = 0
            for j in range (sample.get_network_size()):
                if (i == j):
                    continue
                cumulativeDelay = S[i,j]["AggInfo"]["AvgDelay"] + cumulativeDelay
                cumulativeJitter = S[i,j]["AggInfo"]["Jitter"] + cumulativeJitter
            #Node i
            delays_lst.append(cumulativeDelay)
            jitter_lst.append(cumulativeJitter)

        aux_lst = [delays_lst,jitter_lst]
        metricas = np.asarray(aux_lst)
        input_to_tensor = torch.Tensor(metricas)
        in_data.append(input_to_tensor)
        print("in_data: ", in_data)
        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        P = sample.get_performance_matrix()
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
    
    #  return {"traffic": np.expand_dims(list(nx.get_node_attributes(HG, 'traffic').values()), axis=1),
    #             "packets": np.expand_dims(list(nx.get_node_attributes(HG, 'packets').values()), axis=1),
    #             "length": list(nx.get_node_attributes(HG, 'length').values()),
    #             "model": list(nx.get_node_attributes(HG, 'model').values()),
    #             "eq_lambda": np.expand_dims(list(nx.get_node_attributes(HG, 'eq_lambda').values()), axis=1),
    #             "avg_pkts_lambda": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_pkts_lambda').values()), axis=1),
    #             "exp_max_factor": np.expand_dims(list(nx.get_node_attributes(HG, 'exp_max_factor').values()), axis=1),
    #             "pkts_lambda_on": np.expand_dims(list(nx.get_node_attributes(HG, 'pkts_lambda_on').values()), axis=1),
    #             "avg_t_off": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_t_off').values()), axis=1),
    #             "avg_t_on": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_t_on').values()), axis=1),
    #             "ar_a": np.expand_dims(list(nx.get_node_attributes(HG, 'ar_a').values()), axis=1),
    #             "sigma": np.expand_dims(list(nx.get_node_attributes(HG, 'sigma').values()), axis=1),
    #             "capacity": np.expand_dims(list(nx.get_node_attributes(HG, 'capacity').values()), axis=1),
    #             "queue_size": np.expand_dims(list(nx.get_node_attributes(HG, 'queue_size').values()), axis=1),
    #             "policy": list(nx.get_node_attributes(HG, 'policy').values()),
    #             "priority": list(nx.get_node_attributes(HG, 'priority').values()),
    #             "weight": np.expand_dims(list(nx.get_node_attributes(HG, 'weight').values()), axis=1),
    #             "delay": list(nx.get_node_attributes(HG, 'delay').values()),
    #             "link_to_path": tf.ragged.constant(link_to_path),
    #             "queue_to_path": tf.ragged.constant(queue_to_path),
    #             "queue_to_link": tf.ragged.constant(queue_to_link),
    #             "path_to_queue": tf.ragged.constant(path_to_queue, ragged_rank=1),
    #             "path_to_link": tf.ragged.constant(path_to_link, ragged_rank=1)
    #             }, list(nx.get_node_attributes(HG, 'delay').values())
