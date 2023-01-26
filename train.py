#Core import
from preparacion_dataset import preparation_dataset, prepare_data

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

# Normalize data
# https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html#torch-nn-functional-normalize

metricas_entrada = F.normalize(metricas_entrada)

#Reshape data in order to fulfill specified shape
input = metricas_entrada[:,:2,:]
labels =metricas_entrada[:,2,:]

input = np.reshape(input, (20, 9, 2))
labels = np.reshape(labels, (20, 9, 1))
#comparador = metricas_salida[0:4]

# Normaliza datos de entrada y de salida
# https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html#torch-nn-functional-normalize

data = prepare_data(input=input, edge_index=edge_index, labels=labels)
#Select number of epoch
epoch = 100


# JORGE: fíjate en un solo caso, es decir, no vayas cambiando de grafos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyGCN().to(device)
model.train(True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(epoch):
    
    print("Epoch: ", i+1)

    testloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)

    for j, data_in in enumerate(testloader):
        optimizer.zero_grad()
        print(f'iteracion: {j}')
        out = model(x = data_in.x , edge_index=data_in.edge_index)
        loss = F.mse_loss(out, data_in.y)

        # prediction = model.forward(data, input_edge_tensor)
        #print(prediction)
        #loss = torch.sqrt(F.mse_loss(prediction, comparador))
        # loss = F.mse_loss(prediction, comparador)

        print("Loss: ", loss)
        loss.backward()
        optimizer.step()
    
    #model.eval()
