import torch
import matplotlib.pyplot as plt
import numpy as np

def cdf_plot(tensor, name):
    
    sorted_tensor, indices = torch.sort(tensor)
    x = torch.arange(1, len(tensor) + 1) / len(tensor)

    fig, ax = plt.subplots()

    plt.plot(x,sorted_tensor)
    plt.title(name+" CDF")
    if name == "Delay":
        plt.ylabel("Delay [s]")
    if name == "Packets_Generated":
        plt.ylabel("Packets Generated")
    if name == "Jitter":
        plt.ylabel("Jitter")

    plt.savefig('cdf_plots/' +name+ '.png')

def cdf_hist (tensor):
    fig, ax = plt.subplots()
    sorted_tensor, indices = torch.sort(tensor)
    np.cumsum(sorted_tensor)
    x = torch.arange(1, len(tensor) + 1) / len(tensor)
    plt.plot(x,sorted_tensor)
   # plt.hist(tensor, cumulative=True, label='CDF', histtype='step', alpha=0.8)
    
    plt.savefig('cdf_plots/Delay_HIST_CDF.png')
