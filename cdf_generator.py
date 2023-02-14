import torch
import matplotlib.pyplot as plt

def cdf_plot(tensor, name):
    
    sorted_tensor, indices = torch.sort(tensor)
    y = torch.arange(1, len(tensor) + 1) / len(tensor)

    fig, ax = plt.subplots()
    plt.plot(y,sorted_tensor)
    plt.savefig('cdf_plots/' +name+ '.png')
