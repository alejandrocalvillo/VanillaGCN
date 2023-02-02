from VanillaGCN import datanetAPI
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from astropy.visualization import hist

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

# Plot histogram of the delay of all path of a sample
# W select a random sample (Or we can try to chose one!)
# s samples_lst[0]
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
path = 'histograms/hist'
for i in range(60):
    path+repr(i+1)+".png"
    if os.path.exists(path):
        continue
    else:
        plt.savefig(path+repr(i+1)+".png")
        break
plt.close()
