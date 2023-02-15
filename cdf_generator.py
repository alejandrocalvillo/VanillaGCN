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
    fig, ax = plt.subplots
    cdf = np.cumsum(tensor) / np.sum(tensor)

    # Visualizar la CDF
    plt.plot(cdf)
    plt.title('CDF')
    plt.xlabel('Índex')
    plt.ylabel('Delay Accumulated')
        
    plt.savefig('cdf_plots/Delay_HIST_CDF.png')
