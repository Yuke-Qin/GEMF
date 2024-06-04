from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from torch_geometric.utils import softmax
import torch
import torch.nn.functional as F    
from torch_geometric.nn import global_max_pool
from utils import *


class EmbeddingLayer(nn.Module):
    def __init__(self, atom_in_dim, bond_in_dim, rbf_num, hidden_dim):
        super(EmbeddingLayer, self).__init__()
        self.atom_embed = nn.Linear(atom_in_dim, hidden_dim)
        self.bond_embed = nn.Linear(bond_in_dim + rbf_num, hidden_dim)
        self.angle_embed = nn.Linear(rbf_num, hidden_dim)
        self.dist_embed = nn.Linear(rbf_num, hidden_dim)

    def forward(self, x, edge_attr, dist_cov_rbf, dist_ncov_rbf, angle_rbf):
        hx_0 = self.atom_embed(x)
        he_cov_0 = self.bond_embed(torch.cat([edge_attr, dist_cov_rbf], dim = -1))
        ha_0 = self.angle_embed(angle_rbf)
        he_ncov_0 = self.dist_embed(dist_ncov_rbf)
        return hx_0, he_cov_0, ha_0, he_ncov_0


class NodeGraphLayer(MessagePassing): 
    def __init__(self, hidden_dim, num_heads, dropout):
        super(NodeGraphLayer, self).__init__(aggr = 'add', flow = 'source_to_target')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dm_h = hidden_dim // num_heads 
        self.attn_dropout = dropout
        self.attn_w = nn.Linear(2 * self.dm_h, 1)
        self.attn_activatoion = nn.LeakyReLU()
        self.combine = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_attr):   
        out = self.propagate(edge_index = edge_index, x = x, edge_attr = edge_attr)
        hx_out = self.combine(torch.cat([x, out], dim = -1))
        return  hx_out


    # 使用多头注意机制聚合邻域消息
    def message(self, x_j, x_i, edge_index_i, edge_attr):
        x_j = x_j * edge_attr
        x_i, x_j = x_i.view(-1, self.num_heads, self.dm_h), x_j.view(-1, self.num_heads, self.dm_h)
        
        beta = self.attn_w(torch.cat([x_j, x_i], dim = -1)).permute(1, 0, 2)
        beta = self.attn_activatoion(beta.permute(1, 0, 2))

        alpha = softmax(beta, edge_index_i)
        alpha = F.dropout(alpha, self.attn_dropout, self.training)

        return (alpha * x_j).view(x_j.size(0), -1)

class LineGraphLayer(MessagePassing): 
    def __init__(self, hidden_dim, dropout):
        super(LineGraphLayer, self).__init__(aggr = "add", flow = "source_to_target")
        self.hidden_dim = hidden_dim
        self.A1 = nn.Linear(hidden_dim, hidden_dim)
        self.A2 = nn.Linear(hidden_dim, hidden_dim)
        self.A3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.combine = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_attr): 
        identity_e = edge_attr
        out = self.propagate(edge_index = edge_index, x = x, edge_attr = edge_attr)
        hx_out = self.combine(torch.cat([x, out], dim = -1))

        source, target = edge_index[0], edge_index[1]
        eta = self.A1(x[source]) + self.A2(x[target]) + self.A3(edge_attr)
        he_out = identity_e + self.dropout(self.relu(eta))
        return hx_out, he_out

    def message(self, x_j, edge_attr): 
        return x_j * edge_attr


class CovalentLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(CovalentLayer, self).__init__()
        self.node2edge = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU())
        self.edge2edge = LineGraphLayer(hidden_dim, dropout)
        self.edge2node = NodeGraphLayer(hidden_dim, num_heads, dropout)


    def forward(self, x, edge_index, edge_attr, bond_edge_index, bond_edge_attr):
    #def forward(self, x, edge_index, edge_attr):
        source, target = edge_index[0], edge_index[1]
        he_0 = self.node2edge(torch.cat([x[source], x[target], edge_attr], dim = -1))
        he, ha = self.edge2edge(he_0, bond_edge_index, bond_edge_attr)
        hx = self.edge2node(x, edge_index, he)
        return hx, he, ha


class NonCovalentLayer(MessagePassing):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(NonCovalentLayer, self).__init__()
        self.hidden_dim = hidden_dim      

        self.B1 = nn.Linear(hidden_dim, hidden_dim)
        self.B2 = nn.Linear(hidden_dim, hidden_dim)
        self.B3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.edge2node = NodeGraphLayer(hidden_dim, num_heads, dropout)

    def forward(self, x, edge_index_inter, edge_attr_inter):     
        identity_e = edge_attr_inter
        hx_out = self.edge2node(x, edge_index_inter, edge_attr_inter)
        source, target = edge_index_inter[0], edge_index_inter[1]
        eta = self.B1(x[source]) + self.B2(x[target]) + self.B3(edge_attr_inter)
        he_out = identity_e + self.dropout(self.relu(eta))
        return hx_out, he_out


class HeteroInteractionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, lastlayer):
        super(HeteroInteractionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.lastlayer = lastlayer

        self.cov = CovalentLayer(hidden_dim, num_heads, dropout)
        self.ncov = NonCovalentLayer(hidden_dim, num_heads, dropout)
  
        self.I = nn.GRUCell(hidden_dim, hidden_dim)
        if lastlayer == False:      
            self.w1 = nn.Linear(hidden_dim, hidden_dim)
            self.w2 = nn.Linear(hidden_dim, hidden_dim)
            self.w3 = nn.Linear(hidden_dim, hidden_dim)
            self.w4 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hx_cov, edge_index, edge_attr, bond_edge_index, bond_edge_attr, hx_ncov, edge_index_inter, edge_attr_inter):

        hx_cov_out, he_cov, ha = self.cov(hx_cov, edge_index, edge_attr, bond_edge_index, bond_edge_attr)
        hx_ncov_out, he_ncov = self.ncov(hx_ncov, edge_index_inter, edge_attr_inter)
        mutual_info = self.I(hx_cov_out, hx_ncov_out)

        if self.lastlayer == False:    
            g1 = torch.sigmoid(self.w1(hx_cov_out) + self.w2(mutual_info))
            g2 = torch.sigmoid(self.w3(hx_ncov_out) + self.w4(mutual_info))
            hx_cov_in = (1 - g1) * hx_cov_out + g1 * mutual_info
            hx_ncov_in = (1 - g2) * hx_ncov_out + g2 * mutual_info

            return hx_cov_in, he_cov, ha, hx_ncov_in, he_ncov, mutual_info
        
        else:
            return hx_cov_out, hx_ncov_out, mutual_info


class PredictLayer(nn.Module):
    def __init__(self, atom_dim, hidden_dim_list, dropout):
        super(PredictLayer, self).__init__()
        self.mlp = nn.ModuleList()
        for hidden_dim in hidden_dim_list:
            self.mlp.append(nn.Linear(atom_dim, hidden_dim))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))
            atom_dim = hidden_dim
        self.output_layer = nn.Linear(atom_dim, 1)
    
    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        output = self.output_layer(x)
        return output
