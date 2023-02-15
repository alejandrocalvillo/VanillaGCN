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

    # Iteramos sobre cada grafo en el tensor
    for i in range(tensor.shape[0]):
        # Seleccionamos los datos correspondientes al grafo actual
        datos = tensor[i, :, :].reshape(-1, 1)

        # Ordenamos los datos de forma ascendente
        datos_ordenados = np.sort(datos, axis=0)

        # Calculamos la fracción acumulada de observaciones
        cdf = np.cumsum(np.ones_like(datos_ordenados)).astype(np.float64) / datos_ordenados.shape[0]

        # Traza la CDF en el mismo gráfico
        plt.plot(datos_ordenados, cdf, label=f"Grafo {i+1}")

    # Etiquetas y leyenda
    plt.xlabel('Delay')
    plt.ylabel('Cumulative Probability')

    plt.title('CDF')
    plt.xlabel('Delay')
    plt.ylabel('Cumulative Probability')
    plt.savefig('cdf_plots/Delay_HIST_CDF_Separated.png')
