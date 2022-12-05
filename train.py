from preparacion_dataset import preparation_dataset, hg_to_data
import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, Sequential

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

a = edge_index[0].todense()
a_tensor = torch.Tensor(a)
print("A:",a_tensor)
print("A shape: ", a_tensor.shape)
print("--------------------------------------------")

model = Sequential('x, edge_index', [
    (GCNConv(2, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (GCNConv(64, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    Linear(64, 1),
])

model.forward()
