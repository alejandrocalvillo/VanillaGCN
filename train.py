from preparacion_dataset import preparation_dataset, hg_to_data
import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, Sequential

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from model import train_node_classifier, print_results, MyGCN


data_folder_name = "training"
src_path = f"{data_folder_name}/results/dataset1/"
data_folder_name = "checkpoint"
CHECKPOINT_PATH = f"{data_folder_name}/checkpoint1"
metricas_entrada, metricas_salida,edge_index = preparation_dataset(src_path)

print("El tensor de entrada es: ", metricas_entrada)
print("Su forma: ", metricas_entrada.shape)
print("El tensor de salida es: ", metricas_salida)
print("Su forma: ", metricas_salida.shape)
print("Matrix adjacency: ", edge_index[0])
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
edge_tensor = torch.Tensor(a)
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










device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyGCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#train

model.train()
for epoch in range(20):
    optimizer.zero_grad()
    out = model(x = metricas_entrada[0], edge_index=edge_tensor[0])
    loss = F.nll_loss(out, metricas_salida[0])
    loss.backward()
    optimizer.step()