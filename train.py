#Core import
from preparacion_dataset import preparation_dataset, prepare_data, plot_mse_epoch

#GCN Model
from model import MyGCN

#Pytorch 
import torch

#Pytorch Functional
import torch.nn.functional as F

#Numpy
import numpy as np

#Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load data from BCN-GNN-CHALLENGE

data_folder_name = "training"
src_path = f"{data_folder_name}/results/dataset1/"
# data_folder_name = "checkpoint"
# CHECKPOINT_PATH = f"{data_folder_name}/checkpoint1"

#Load the data
metricas_entrada,edge_index = preparation_dataset(src_path)

# Normalize data
# https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html#torch-nn-functional-normalize

metricas_entrada = F.normalize(metricas_entrada)

#Select features to predict
input = metricas_entrada[:,:2,:] #train
labels =metricas_entrada[:,2,:] #train

#Reshape data in order to fulfill specified shape
input = np.reshape(input, (20, 9, 2))
labels = np.reshape(labels, (20, 9, 1))

print(input)
print(labels)
#Prepare the dataset
data = prepare_data(input=input, edge_index=edge_index, labels=labels)


class SimpleCustomBatch:
    def __init__(self, data):
        data = prepare_data(input=input, edge_index=edge_index, labels=labels)
        self.inp = data.x
        self.tgt = data.y

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

dataset = torch.utils.data.TensorDataset(data.x, data.y)


#Select number of epoch
epoch = 10000
lr = 1e-6

# JORGE: fíjate en un solo caso, es decir, no vayas cambiando de grafos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyGCN().to(device)
model.train(True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
iterations = 0
loss_ar = []
for i in range(epoch):
    
    print("Epoch: ", i+1)

    testloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn= collate_wrapper, shuffle=True)

    for j, data_in in enumerate(testloader):
        optimizer.zero_grad()
        iterations = iterations + 1
        print(f'iteracion: {j}')
        out = model(x = data_in.inp , edge_index=data.edge_index)
        loss = F.mse_loss(out, data_in.tgt)
        mse_loss = loss.detach().numpy()
        loss_ar.append(mse_loss)
        print("Loss: ", loss)
        loss.backward()
        optimizer.step()

state_dict = model.state_dict()
torch.save(state_dict, 'weigths/model_weights.pt')
plot_mse_epoch(iterations, loss_ar, epoch, lr)
