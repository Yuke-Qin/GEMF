from torch_geometric.nn.conv import GCNConv,GATConv
from torch_geometric.nn.glob.gmt import*
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_dense_batch
from layers import *
from utils import *


class Mymodel(nn.Module):
    def __init__(self, atom_in_dim, bond_in_dim, rbf_num, hidden_dim, num_heads,  dropout):
        super(Mymodel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.embed_layer = EmbeddingLayer(atom_in_dim, bond_in_dim, rbf_num, hidden_dim)

        self.hetero_interaction1 = HeteroInteractionLayer(hidden_dim, num_heads, dropout, lastlayer = False)
        self.hetero_interaction2 = HeteroInteractionLayer(hidden_dim, num_heads, dropout, lastlayer = False)
        self.hetero_interaction3 = HeteroInteractionLayer(hidden_dim, num_heads, dropout, lastlayer = True)

        self.hetero_fusion = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout())
        self.predict_layer = PredictLayer(atom_dim = hidden_dim, hidden_dim_list = [512, 256, 128], dropout = dropout)


    def forward(self, data):
        x, batch, pos = data.x, data.batch, data.pos,
        edge_index, edge_index_inter, edge_attr = data.edge_index, data.edge_index_inter, data.edge_attr
        bond_edge_index, bond_edge_attr = data.bond_edge_index, data.bond_edge_attr

        source_cov, target_cov = edge_index 
        pos_vector_cov = pos[source_cov] - pos[target_cov]
        source_ncov, target_ncov = edge_index_inter 
        pos_vector_ncov = pos[source_ncov] - pos[target_ncov]

        dist_cov_rbf = D_rbf(torch.norm(pos_vector_cov, dim = -1), 0, 6, 9, device = x.device)
        dist_ncov_rbf = D_rbf(torch.norm(pos_vector_ncov, dim = -1), 0, 6, 9, device = x.device)
        angle_rbf = A_rbf(bond_edge_attr, 9, device = x.device) 
        
        hx_0, he_cov_0, ha_0, he_ncov_0 = self.embed_layer(x, edge_attr, dist_cov_rbf, dist_ncov_rbf, angle_rbf) 
        
        hx_cov_1, he_cov_1, ha_1, hx_ncov_1, he_ncov_1, mutual_info_1 = self.hetero_interaction1(hx_0, edge_index, he_cov_0, bond_edge_index, ha_0, hx_0, edge_index_inter, he_ncov_0)
        hx_cov_2, he_cov_2, ha_2, hx_ncov_2, he_ncov_2, mutual_info_2 = self.hetero_interaction2(hx_cov_1, edge_index, he_cov_1, bond_edge_index, ha_1, hx_ncov_1, edge_index_inter, he_ncov_1)
        hx_cov_3, hx_ncov_3, mutual_info_3 = self.hetero_interaction3(hx_cov_2, edge_index, he_cov_2, bond_edge_index, ha_2, hx_ncov_2, edge_index_inter, he_ncov_2)
     
        hx_fusion = self.hetero_fusion(torch.cat([mutual_info_1, mutual_info_2, mutual_info_3], dim = -1))
        hx_fusion_pool = global_add_pool(hx_fusion, batch)
        hx_cov_pool = global_add_pool(hx_cov_3, batch)
        hx_ncov_pool = global_add_pool(hx_ncov_3, batch)

        output = self.predict_layer(hx_fusion_pool + hx_cov_pool + hx_ncov_pool)
        return output.view(-1)


