from preparacion_dataset import preparation_dataset, hg_to_data
import torch
from torch_geometric.nn import GCNConv

data_folder_name = "training"
src_path = f"{data_folder_name}/results/dataset1/"

HG = preparation_dataset(src_path)
X, y_t, edge_index = hg_to_data(HG)

train_examples = []
for item in X[0]:
    tensor = torch.Tensor(X[0].get(item))
    train_examples.append(tensor)
in_data = torch.stack(train_examples)



c1= GCNConv(len(in_data),len(y_t))
y_pred = c1.forward(x=in_data, edge_index=edge_index[0])