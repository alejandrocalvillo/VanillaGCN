import torch
import matplotlib.pyplot as plt

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
