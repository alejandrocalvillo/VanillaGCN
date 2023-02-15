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

   # Seleccionar la feature de interés (la última columna del tensor)
    feature = tensor[:, :, -1]

    # Iterar sobre el primer índice del tensor para calcular la CDF de cada grafo
    for i in range(tensor.shape[0]):
        # Seleccionar la feature de interés para el grafo actual
        feature_i = feature[i, :].ravel()
        
        # Calcular la suma acumulada de la feature
        cumulative_sum = torch.cumsum(feature_i, dim=0)
        
        # Calcular la CDF de la feature
        cdf = cumulative_sum / torch.sum(feature_i)
        
        # Visualizar la CDF con un color diferente para cada grafo
        plt.plot(cdf, color=plt.cm.Set1(i), label=f"Grafo {i+1}")
        
    # Configurar el plot y mostrarlo
    plt.title('CDF')
    plt.xlabel('Índex')
    plt.ylabel('Delay Accumulated')
    
    plt.savefig('cdf_plots/Delay_HIST_CDF.png')
