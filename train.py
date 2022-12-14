#Core import
from preparacion_dataset import preparation_dataset, data_creator

#GCN Model
from model import MyGCN

#Pytorch 
import torch

#Pytorch Functional
import torch.nn.functional as F


#Numpy
import numpy as np

#Load data from BCN-GNN-CHALLENGE

data_folder_name = "training"
src_path = f"{data_folder_name}/results/dataset1/"
data_folder_name = "checkpoint"
CHECKPOINT_PATH = f"{data_folder_name}/checkpoint1"
metricas_entrada, metricas_salida,edge_index = preparation_dataset(src_path)

#Reshape data in order to fulfill specified shape
metricas_entrada = np.reshape(metricas_entrada, (20, 9, 2))
metricas_salida = np.reshape(metricas_salida, (20, 9, 1))
comparador = metricas_salida[0:4]


#Select number of epoch

epoch = 200

for i in range(epoch):
    
    print("Epoch: ", i)
    data = 0
    #Take Adjacency_Matrix and Reshape it in order to fulfill specified shape
    j = 0 + i
    if i >= 19:
        j = 0
    else:
        a = edge_index[j].todense()
        edge_tensor = torch.tensor(a, dtype = torch.long)
        input_edge_tensor = edge_tensor.nonzero().t().contiguous()
        #data = data_creator(metricas_entrada,metricas_salida,input_edge_tensor)

    testloader = torch.utils.data.DataLoader(metricas_entrada, batch_size=4, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyGCN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    #train
    print(comparador)
    model.train()
    for data in testloader:
        optimizer.zero_grad()
        print(f'iteracion, data.size={data.size()}')
        out = model(x = data , edge_index=input_edge_tensor)
        loss = F.mse_loss(out, comparador)
        loss.backward()
        print("Loss: ", loss)
        optimizer.step()
    
    model.eval()
