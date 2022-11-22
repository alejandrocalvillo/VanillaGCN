from VanillaGCN import datanetAPI
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from astropy.visualization import hist
def main():
    data_folder_name = "training"
    src_path = f"{data_folder_name}/results/dataset1/"
    # Range of the maximum average lambda | traffic intensity used 
    #  max_avg_lambda_range = [min_value,max_value] 
    max_avg_lambda_range = [10,10000]
    # List of the network topology sizes to use
    net_size_lst = [4,5,6,7,8,9,10]
    # Obtain all the samples from the dataset
    reader = datanetAPI.DatanetAPI(src_path,max_avg_lambda_range, net_size_lst)
    samples_lst = []
    graph_topology = []
    for sample in reader:
        samples_lst.append(sample)
        G = nx.DiGraph(sample.get_topology_object())
    print ("Number of selected samples: ",len(samples_lst))

    A = nx.adjacency_matrix(G[0])
    print(A)