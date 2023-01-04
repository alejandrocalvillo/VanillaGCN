#Core import
from preparacion_dataset import preparation_dataset, hg_to_data

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


a = edge_index[0].todense()
edge_tensor = torch.tensor(a, dtype = torch.long)
input_edge_tensor = edge_tensor.nonzero().t().contiguous()


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
    print(out)
    print(out.size())
    print(comparador[:,0,:])
    print(comparador[:,0,:].size())
    loss = F.mse_loss(out, comparador)
    loss.backward()
    optimizer.step()
