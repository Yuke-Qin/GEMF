import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Batch, Data
import re, os
from scipy.spatial import distance_matrix
from torch_geometric.utils import add_self_loops, remove_isolated_nodes, to_networkx

import networkx as nx
from scipy import sparse as sp
from utils import *
torch.set_printoptions(profile='full')

atom_type = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I','other']   

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def calc_atom_features(atom, explicit_H = False):
    atom_feats = one_of_k_encoding_unk(atom.GetSymbol(),atom_type) +\
                 one_of_k_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5,6]) +\
                 one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6]) +\
                 one_of_k_encoding_unk(atom.GetHybridization(), 
                                       [
                                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                                        Chem.rdchem.HybridizationType.SP3D2
                                       ]) + [atom.GetIsAromatic()]
                 
    if not explicit_H: # 不显示氢原子
        atom_feats  = atom_feats +  one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])   
    return atom_feats


def calc_covalentBond_features(bond, use_chirality = False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,\
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,\
        bond.GetIsConjugated(), bond.IsInRing()
    ]

    if use_chirality:
        bond_feats += one_of_k_encoding_unk(
            str(bond.GetStereo()), 
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
    return torch.FloatTensor(bond_feats)


def get_atom_features(mol, graph, explicit_H = False, use_chirality = False):
    num_atoms = mol.GetNumAtoms()
    atom_feats = np.array([calc_atom_features(atom = a, explicit_H = explicit_H ) for a in mol.GetAtoms()], dtype = np.float32)
    atom_nodes = [atom.GetIdx() for atom in mol.GetAtoms()]

    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True, useLegacyImplementation = False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis = 1)   
    atom_feats = torch.FloatTensor(atom_feats)
    
    atom_withattr_list = []
    for node in atom_nodes:
        atom_withattr_list.append((node, {'feats':atom_feats[node]}))
    graph.add_nodes_from(atom_withattr_list)


def get_covalent_features(mol, graph, use_chirality = False):
    for bond in mol.GetBonds():
        bond_feats = calc_covalentBond_features(bond = bond, use_chirality = use_chirality)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        graph.add_edge(i, j, feats = bond_feats)


def load_mol(molpath, explicit_H=False, use_chirality = False):
	# load mol
    if re.search(r'.pdb$', molpath):
        mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
    elif re.search(r'.mol2$', molpath):
        mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
    elif re.search(r'.sdf$', molpath):			
        mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")	
    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def gen_mol2graph(mol, explicit_H, use_chirality):
    graph = nx.Graph()
    get_atom_features(mol, graph, explicit_H = explicit_H, use_chirality = use_chirality)
    get_covalent_features(mol, graph, use_chirality = use_chirality)
    graph = graph.to_directed()

    x = torch.stack([feat['feats'] for node, feat in graph.nodes(data = True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data = False)]).T
    edge_attr = torch.stack([edge_feats['feats'] for _, _, edge_feats in graph.edges(data = True)]) 

    #construct line graph
    edge_list = [[u, v] for (u, v) in graph.edges(data = False)]
    num_atoms = mol.GetNumAtoms()
    num_bonds = len(edge_list)
    bond_to_atom = np.zeros((num_bonds, num_atoms), dtype=np.int64)
    atom_to_bond = np.zeros((num_atoms, num_bonds), dtype=np.int64)

    for idx, edge in enumerate(edge_list):
        bond_to_atom[idx, edge[1]] = 1
        atom_to_bond[edge[0], idx] = 1
   
    bondgraph_adj = bond_to_atom @ atom_to_bond
    np.fill_diagonal(bondgraph_adj, 0)
    bondgraph_adj[range(num_bonds), [edge_list.index([e[1],e[0]]) for e in edge_list]] = 0
    src, dst = np.where(bondgraph_adj > 0)

    num_edges = len(src)
    bond_edge_index = [[src[i], dst[i]] for i in range(num_edges)]
    bond_angles = np.zeros(num_edges)
    coord = mol.GetConformers()[0].GetPositions()

    for idx, edge in enumerate(bond_edge_index):
        vec1 = coord[edge_index[0][edge[0]]] - coord[edge_index[1][edge[0]]]
        vec2 = coord[edge_index[0][edge[1]]] - coord[edge_index[1][edge[1]]]

        assert (np.linalg.norm(vec1) > 0 ) and (np.linalg.norm(vec2) > 0 )

        cos = (-np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        bond_angles[idx] = np.degrees(np.arccos(cos))

    bond_edge_attr = torch.FloatTensor(bond_angles)
    bond_edge_index = torch.LongTensor(bond_edge_index).T

    return x, edge_index, edge_attr, bond_edge_index, bond_edge_attr


def gen_inter_graph(ligand, pocket, cutoff):
    atom_num_l = ligand.GetNumAtoms()   
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()      
    pos = torch.concat([torch.FloatTensor(pos_l), torch.FloatTensor(pos_p)], dim=0)
    
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < cutoff)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j + atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data = False)]).T

    return edge_index_inter, pos


def gen_graph(data_root, complex_name, cutoff):
    complex_path = os.path.join(data_root, complex_name)
    ligand_path = os.path.join(complex_path + '/', "{}_ligand.pdb".format(complex_name))
    pocket_path = os.path.join(complex_path + '/', "{}_pocket{}A.pdb".format(complex_name, cutoff))

    ligand = load_mol(ligand_path, explicit_H = False, use_chirality = False)
    pocket = load_mol(pocket_path, explicit_H = False, use_chirality = False)

    if (ligand == None) or (pocket == None):
        return None
    
    # construct intra_graph
    x_l, edge_index_l, edge_attr_l, bond_edge_index_l, bond_edge_attr_l = gen_mol2graph(ligand, explicit_H = False, use_chirality = False)
    x_p, edge_index_p, edge_attr_p, bond_edge_index_p, bond_edge_attr_p = gen_mol2graph(pocket, explicit_H = False, use_chirality = False)

    x = torch.cat([x_l, x_p], dim = 0)
    atom_num_l = len(x_l)
    atom_num_p = len(x_p)
    edge_index = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim = -1)
    edge_attr = torch.cat([edge_attr_l, edge_attr_p], dim = 0)
    bond_num_l = len(edge_index_l)
    bond_edge_index = torch.cat([bond_edge_index_l, bond_edge_index_p + bond_num_l], dim = -1)
    bond_edge_attr = torch.cat([bond_edge_attr_l, bond_edge_attr_p], dim = 0)

    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)

    # construct inter_graph
    edge_index_inter, pos = gen_inter_graph(ligand, pocket, cutoff)
    graph = Data(x = x, edge_index = edge_index,  edge_attr = edge_attr,\
                 bond_edge_index = bond_edge_index, bond_edge_attr = bond_edge_attr,\
                 edge_index_inter = edge_index_inter, pos = pos, split = split)
    
    return graph


if __name__ == '__main__':

    cutoff = 5
    data_root = './protein_ligand_affinity/data/general-set/'
    pkl_root = './protein_ligand_affinity/data/graph/'
    csv_root = './protein_ligand_affinity/data/' 

    for phase in ['test2016']:
        print("doing processing {}_set!".format(phase))
        csv_path = os.path.join(csv_root, "{}.csv".format(phase))
        save_path = os.path.join(pkl_root, "{}_{}A.pkl".format(phase, cutoff))

        df = pd.read_csv(csv_path)
        pdbid_name =  df['pdbid'].to_list()

        pka_dict = {}
        for i, row in df.iterrows():
            pka_dict[row['pdbid']] = row['-logKd/Ki']
        graph_list = []

        bad_complex_list = []
        for pdbid in pdbid_name:
            graph = gen_graph(data_root, pdbid, cutoff)
            if graph == None:
                bad_complex_list.append(pdbid)
                continue
            
            graph.y = pka_dict[pdbid]
            graph_list.append(graph)

        torch.save(graph_list, save_path)
        print(bad_complex_list)
