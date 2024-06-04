import os
import torch
import torch.nn as nn
import random
import numpy as np
from torch_geometric.loader import DataLoader

from sklearn.metrics import mean_squared_error
from dataset import MyDataset
from utils import *
from metrics import *
import matplotlib.pyplot as plt
from model import Mymodel

BATCH_SIZE = 128
LR = 5e-4
EPOCH = 200

def train(model, device, train_loader): 
    model.train()     
    epoch_loss = 0
    pred_list = []
    label_list = []
    for batch_idx, data in enumerate(train_loader):   
        data = data.to(device) 
        label = data.y
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        pred_list.append(out.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy()) 
    y_pred = np.concatenate(pred_list, axis=0)
    y_true = np.concatenate(label_list, axis=0)    
    epoch_rmse = RMSE(y_true, y_pred)
    epoch_loss /= (batch_idx + 1)

    return epoch_loss, epoch_rmse


def valid_and_test(model, device, valid_loader):
    model.eval()
    pred_list = []
    label_list = []
    for _, data in enumerate(valid_loader):
        data = data.to(device) 
        with torch.no_grad():
            label = data.y
            pred = model(data)
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
        
    y_pred = np.concatenate(pred_list, axis=0)
    y_true = np.concatenate(label_list, axis=0) 

    rmse = RMSE(y_true, y_pred)
    mae = MAE(y_true, y_pred)
    r = PR(y_true, y_pred)
    sd = SD(y_true,y_pred)

    return rmse, mae, r, sd


if __name__ == "__main__":
    cutoff = 5
    root = 'protein_ligand_affinity/data/'
    dataset = 'general'

    train_pkl = './protein_ligand_affinity/data/graph/train_{}A.pkl'.format(cutoff)
    valid_pkl = './protein_ligand_affinity/data/graph/valid_{}A.pkl'.format(cutoff) 
    casf2016_pkl = './protein_ligand_affinity/data/graph/test_{}A.pkl'.format(cutoff)
    casf2013_pkl = './protein_ligand_affinity/data/graph/CASF2013_coreset_{}A.pkl'.format(cutoff)
    test2019_pkl = './protein_ligand_affinity/data/graph/test2019_{}A.pkl'.format(cutoff)

    train_data = torch.load(train_pkl)
    valid_data = torch.load(valid_pkl)
    test2016_data = torch.load(casf2016_pkl)
    test2013_data = torch.load(casf2013_pkl)
    test2019_data = torch.load(test2019_pkl)

    train_data = MyDataset(root, dataset, train_data, 'train')
    valid_data = MyDataset(root, dataset, valid_data, 'valid')
    test2016_data = MyDataset(root, dataset, test2016_data, 'test')
    test2013_data = MyDataset(root, dataset, test2013_data, 'casf2013')
    test2019_data = MyDataset(root, dataset, test2019_data, 'test2019')

    repeats = 3
    for repeat in range(repeats):

        train_loader = DataLoader(train_data, BATCH_SIZE, shuffle = True)
        valid_loader = DataLoader(valid_data, BATCH_SIZE)
        test2016_loader = DataLoader(test2016_data, BATCH_SIZE)
        test2013_loader = DataLoader(test2013_data, BATCH_SIZE)
        test2019_loader = DataLoader(test2019_data, BATCH_SIZE)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Mymodel(atom_in_dim = 35, bond_in_dim = 6, hidden_dim = 256, rbf_num = 9, num_heads = 4, dropout = 0.1)
        model = model.to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = 1e-6)

        best_v_rmse = 1.3
        best_model_list = []

        for epoch in range(EPOCH):

            # train
            train_epoch_loss, train_epoch_rmse = train(model, device, train_loader)
            print("train in epoch: {}, epoch_loss: {:.6f}, train_rmse: {:.6f}".format(epoch + 1, train_epoch_loss, train_epoch_rmse))

            # validation

            v_rmse, v_mae, v_r, v_sd = valid_and_test(model, device, valid_loader)
            print("validation in epoch: {}, rmse: {:.6f}, mae: {:.6f}, r: {:.6f}, sd: {:.6f}".format(epoch + 1, v_rmse, v_mae, v_r, v_sd))

            if  v_rmse < best_v_rmse:
                best_v_rmse = v_rmse
                early_stop_iter = 0

                model_name = "epoch-%d, valid_rmse-%.6f, valid_mae-%.6f, valid_r-%.6f, valid_sd-%.6f"\
                %(epoch + 1, v_rmse, v_mae, v_r, v_sd)

                model_dir = 'protein_ligand_affinity/data/checkpoint_model{}/'.format(repeat)
                model_path = os.path.join(model_dir, model_name + '.pt')

                best_model_list.append(model_path)
                torch.save(model.state_dict(), model_path)
                print(f"model has been saved to {model_path}")

        # test
        ckpt = best_model_list[-1]
        model.load_state_dict(torch.load(ckpt))
        v_rmse, v_mae, v_r, v_sd = valid_and_test(model, device, valid_loader)
        
        print("doing final testing!")
        print("valid_rmse: {:.6f}, valid_mae: {:.6f}, valid_r: {:.6f}, valid_sd: {:.6f}".format(v_rmse, v_mae, v_r, v_sd))

        t2016_rmse, t2016_mae, t2016_r, t2016_sd = valid_and_test(model, device, test2016_loader)
        print("test_rmse: {:.6f}, test_mae: {:.6f}, test_r: {:.6f}, test_sd: {:.6f}".format(t2016_rmse, t2016_mae, t2016_r, t2016_sd))
        
        t2013_rmse, t2013_mae, t2013_r, t2013_sd = valid_and_test(model, device, test2013_loader)
        print("test2013_rmse: {:.6f}, test_mae: {:.6f}, test_r: {:.6f}, test_sd: {:.6f}".format(t2013_rmse, t2013_mae, t2013_r, t2013_sd))

        t2019_rmse, t2019_mae, t2019_r, t2019_sd = valid_and_test(model, device, test2019_loader)
        print("test2019_rmse: {:.6f}, test_mae: {:.6f}, test_r: {:.6f}, test_sd: {:.6f}".format(t2019_rmse, t2019_mae, t2019_r, t2019_sd))




    




