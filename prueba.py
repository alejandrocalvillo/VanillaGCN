from model2 import train
from preparacion_dataset import preparation_dataset, hg_to_data

import torch

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

print("Empezamos a entrenar")
train(data=graphs)
