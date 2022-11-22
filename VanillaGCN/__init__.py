from sys import stderr
import warnings

warnings.filterwarnings("ignore")
seed_value = 69420
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED'] = str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import matplotlib.pyplot as plt
from astropy.visualization import hist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.utils.convert as pyg_convert

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from .data_generator import input_fn
from .GCN import VanillaGCN
from .datanetAPI import DatanetAPI

def main(train_path, final_evaluation=False, ckpt_dir="./modelCheckpoints"):
    if (not os.path.exists(train_path)):
        print(f"ERROR: the provided training path \"{os.path.abspath(train_path)}\" does not exist!", file=stderr)
        return None
    TEST_PATH = './validation_dataset'
    if (not os.path.exists(TEST_PATH)):
        print("ERROR: Validation dataset not found at the expected location:",
              os.path.abspath(TEST_PATH), file=stderr)
        return None
    LOG_PATH = './logs'
    if (not os.path.exists(LOG_PATH)):
        print("INFO: Logs folder created at ", os.path.abspath(LOG_PATH))
        os.makedirs(LOG_PATH)
    # Check dataset size
    dataset_size = len([0 for _ in input_fn(train_path, shuffle=True)])
    if not dataset_size:
        print(f"ERROR: The dataset has no valid samples!", file=stderr)
        return None
    elif (dataset_size > 100):
        print(f"ERROR: The dataset can only have up to 100 samples (currently has {dataset_size})!", file=stderr)
        return None
    
    ds_train = input_fn(train_path, shuffle=True, training=True)
    # ds_train = ds_train.repeat()
    ds_test = input_fn(TEST_PATH, shuffle=False)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

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
    for sample in reader:
        samples_lst.append(sample)
    print ("Number of selected samples: ",len(samples_lst))
    
    # print("------------------------TRAIN--------------------------")
    # print(ds_train)
    # print("------------------------TEST--------------------------")
    # print(ds_test)

    # Plot histogram of the delay of all path of a sample
# We select a random sample (Or we can try to chose one!)
# s= samples_lst[0]
    s = random.choice(samples_lst)
    delays_lst = []
    performance_matrix = s.get_performance_matrix()
    for i in range (s.get_network_size()):
        for j in range (s.get_network_size()):
            if (i == j):
                continue
            # Append to the list the average delay of the path i,j.
            delays_lst.append(performance_matrix[i,j]["AggInfo"]["AvgDelay"])

    #Plot histogram using astropy to use correct value of bins
    hist(delays_lst, bins='blocks', histtype='stepfilled',alpha=0.2, density=True)
    plt.title("Histogram showing the delay per path")
    plt.xlabel("Delay (s)")
    plt.show()
    plt.close()