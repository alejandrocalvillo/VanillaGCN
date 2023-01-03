from preparacion_dataset import preparation_dataset, hg_to_data
import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, Sequential

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.data as Data

#Numpy
import numpy as np
from model import train_node_classifier, print_results, MyGCN


data_folder_name = "training"
src_path = f"{data_folder_name}/results/dataset1/"
data_folder_name = "checkpoint"
CHECKPOINT_PATH = f"{data_folder_name}/checkpoint1"
metricas_entrada, metricas_salida,edge_index = preparation_dataset(src_path)

#print("El tensor de entrada es: ", metricas_entrada)
#print("Su forma: ", metricas_entrada.shape)
#print("El tensor de salida es: ", metricas_salida)
#print("Su forma: ", metricas_salida.shape)
#print("Matrix adjacency: ", edge_index[0])
train_examples = []
# for item in X[0]:
#     tensor = torch.Tensor(X[0].get(item))
#     train_examples.append(tensor)
# in_data = torch.stack(HG)
graphs = {'x': metricas_entrada,
          'edge_index': edge_index,
          'y': metricas_salida,
          'num_node_features': 2,
          'num_node_classes': 1
          }
a = edge_index[0].todense()
edge_tensor = torch.tensor(a, dtype = torch.long)
input_edge_tensor = edge_tensor.nonzero().t().contiguous()
print(f'edge tensor.size={edge_tensor.size()}')

print("--------------------------------------------")
# print(in_data.shape)
print("--------------------------------------------")

# print("Empezamos a entrenar")
# node_mlp_model, node_mlp_result = train_node_classifier(model_name="MLP",
#                                                         dataset=graphs,
#                                                         CHECKPOINT_PATH=CHECKPOINT_PATH,
#                                                         c_hidden=16,
#                                                         num_layers=2,
#                                                         dp_rate=0.1)

# print_results(node_mlp_result)

print(edge_tensor)
print("--------------------------------")
print(input_edge_tensor)

metricas_entrada = np.reshape(metricas_entrada, (20, 9, 2))
metricas_salida = np.reshape(metricas_salida, (20, 9, 1))
comparador = metricas_salida[0:4]
print(metricas_entrada[0].size())
print("---------------------------------")
print(metricas_salida[0].size())
print(comparador.size())

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
    loss = F.nll_loss(out, comparador[:,0,:].long())
    loss.backward()
    optimizer.step()
