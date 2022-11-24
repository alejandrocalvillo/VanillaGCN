from preparacion_dataset import preparation_dataset, hg_to_data
from torch_geometric.nn import GCNConv

data_folder_name = "training"
src_path = f"{data_folder_name}/results/dataset1/"

HG = preparation_dataset(src_path)
X, y_t, edge_index = hg_to_data(HG)

c1= GCNConv(len(HG),len(y_t))
y_pred = c1.forward(x=X, edge_index=edge_index)