from preparacion_dataset import preparation_dataset, hg_to_data
import torch
from torch_geometric.nn import GCNConv

data_folder_name = "training"
src_path = f"{data_folder_name}/results/dataset1/"

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



# print(in_data.shape)
print("--------------------------------------------")
c1= GCNConv(2,1)
