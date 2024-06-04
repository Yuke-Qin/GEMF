import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import networkx as nx
from scipy import sparse as sp
import os

def random_split_kfold(datasize, k, seed): # 选定验证集
    np.random.seed(seed)
    indices = list(range(datasize))
    np.random.shuffle(indices)
    split_ratio = 1 / k
    interval = int(split_ratio * datasize)
    train_index = []
    val_index = []
    start = 0
    for i in range(k):
        end = start + interval
        val_index.append(indices[start:end])
        train_index.append(indices[:start] + indices[end:])
        start = end
    return train_index, val_index


def random_split(dataset_size, split_ratio, seed):
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, val_idx = indices[:split], indices[split:]
    return train_idx, val_idx


def D_rbf(D, d_min, d_max, d_count, device = 'cpu'):
    D_mu = torch.linspace(d_min, d_max, d_count).to(device)
    D_mu = D_mu.view([1, -1]) 
    D_sigma = (d_max - d_min) / d_count
    D_expand = torch.unsqueeze(D, -1)
    D_rbf = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return D_rbf

def A_rbf(A, a_count, device = 'cpu'):
    A_mu = torch.linspace(0, math.pi, a_count).to(device)
    A_mu = A_mu.view([1, -1]) # shape = [1, a_count]
    beta = torch.full((1, a_count), math.pow((2 / a_count) * math.pi, -2)).to(device) # shape = [1, a_count]
    A_expand = torch.unsqueeze(A, -1) # shape = [n, 1]
    A_rbf = torch.exp(-beta * (A_expand - A_mu) ** 2)
    return A_rbf


def calc_angle(a, b, c):# 余弦定理求∠C
    cos = (a**2 + b**2 - c**2) / (2 * a * b)
    if cos > 1:
        cos = 1
    if cos < -1:
        cos = -1
    return np.degrees(np.arccos(cos)) # 角度制返回
    
    
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-16)