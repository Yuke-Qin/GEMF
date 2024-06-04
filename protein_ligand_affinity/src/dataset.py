from typing import Any, List, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Dataset, Batch, Data
import os
import pandas as pd
import sys

from utils import random_split


class MyDataset(Dataset):
    def __init__(self, root, dataset, data, phase):
        super().__init__(root)
        self.root = root
        self.dataset = dataset
        self.data = data
        self.phase = phase
        
        self.path = os.path.join(self.processed_dir, self.dataset + '_' + self.phase + '.pt')
        
        if self.path not in self.processed_paths:
            raise FileNotFoundError('File not in processed folder') 

        if os.path.isfile(self.processed_paths[self.processed_paths.index(self.path)]):
            print("Processed is found...")
        else:
            print("Processed data not found, doing processing...")
            self.process()
        self.data = torch.load(self.processed_paths[self.processed_paths.index(self.path)])
            

    @property
    def processed_file_names(self):
        return [self.dataset + '_train.pt', self.dataset + '_valid.pt', self.dataset + '_test.pt', self.dataset + '_casf2013.pt', self.dataset + '_test2019.pt']

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed/')

    def _process(self):
        pass

    def process(self):     
        torch.save(self.data, self.processed_paths[self.processed_paths.index(self.path)])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
