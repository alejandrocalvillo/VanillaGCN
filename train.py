from preparacion_dataset import preparation_dataset
from torch_geometric.nn import GCNConv
data_folder_name = "training"
src_path = f"{data_folder_name}/results/dataset1/"

X, y_t, edge_index = preparation_dataset(src_path)
print(X)

y_nose= GCNConv(X)