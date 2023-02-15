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

    matriz = tensor.reshape(-1, 1)
    matriz_ordenada = np.sort(matriz, axis=0)
    valores, conteos = np.unique(matriz_ordenada, return_counts=True)
    cdf = np.cumsum(conteos).astype(np.float64) / matriz.shape[0]
    plt.plot(valores, cdf)

    plt.title('CDF')
    plt.xlabel('Delay')
    plt.ylabel('Cumulative Probability')
    plt.savefig('cdf_plots/Delay_HIST_CDF.png')
