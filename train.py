#Core import
from preparacion_dataset import preparation_dataset

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
metricas_salida = np.reshape(metricas_salida[:,:,2], (20, 9, 1))
#comparador = metricas_salida[0:4]


print(metricas_entrada.shape)
print(metricas_salida)
print(metricas_salida.shape)

# Normaliza datos de entrada y de salida
# https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html#torch-nn-functional-normalize


#Select number of epoch

epoch = 888888
epoch = 1000


# JORGE: fíjate en un solo caso, es decir, no vayas cambiando de grafos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyGCN().to(device)
model.train(True)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(epoch):
    
    print("Epoch: ", i)
    data = 0
    #Take Adjacency_Matrix and Reshape it in order to fulfill specified shape

    #Podria hacer un random de 1 a 20 
    # JORGE: update the j, it should only be a single graph
    # otherwise, the adjacenecy matrix changes and the learning doesn't hold
    j = 0 + i
    if j >= 19:
        j = 0
        a = edge_index[0].todense()
        edge_tensor = torch.tensor(a, dtype = torch.long)
        input_edge_tensor = edge_tensor.nonzero().t().contiguous()
    else:
        a = edge_index[0].todense()
        edge_tensor = torch.tensor(a, dtype = torch.long)
        input_edge_tensor = edge_tensor.nonzero().t().contiguous()

    #For the Backward Propagation and RMSE not to be compared with the same one 
    coor1 = j
    coor2 = j+4

    if coor2 >= 19:
        coor1 = 0
        coor2 = 4
        comparador = metricas_salida[coor1:coor2]
    else:
        comparador = metricas_salida[coor1:coor2]
        #data = data_creator(metricas_entrada,metricas_salida,input_edge_tensor)

    testloader = torch.utils.data.DataLoader(metricas_entrada, batch_size=4, shuffle=True)


    for data in testloader:
        optimizer.zero_grad()
        print(f'iteracion: {i}, data.size={data.size()}')
        out = model(x = data , edge_index=input_edge_tensor)
        loss = F.mse_loss(out, comparador)

        # prediction = model.forward(data, input_edge_tensor)
        #print(prediction)
        #loss = torch.sqrt(F.mse_loss(prediction, comparador))
        # loss = F.mse_loss(prediction, comparador)

        print("Loss: ", loss)
        loss.backward()
        optimizer.step()
    
    #model.eval()
