import torch
import matplotlib.pyplot as plt

def plot_cdf(tensors, labels):
    sorted_tensors = []
    for tensor in tensors:
        sorted_tensor, indices = torch.sort(tensor)
        sorted_tensors.append(sorted_tensor)
    y = torch.arange(1, len(tensor) + 1) / len(tensor)
    for i, tensor in enumerate(sorted_tensors):
        plt.plot(tensor, y, label=labels[i])
    plt.legend()
    plt.savefig('cdf_plots/CDF_conjunta.png')
