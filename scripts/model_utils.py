import os 
import pandas as pd 
import torch 
import torch.nn as nn 
import numpy as np 
import networkx as nx 

from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import  Dataset
from torch.nn import Linear 
import torch.optim as optim 
from torch_geometric.nn import GCNConv
import time
import tqdm 
import random
import copy

class HARData(InMemoryDataset): 
    """create a dataset object given a
    directory. 
    """
    def __init__(self, dir, TRAIN_PROP=0.7):
        super().__init__()
        data_components = ['edge_list.txt', 'node_attributes.txt', 'node_labels']
        #TODO: check if all component text files are present. 

        # adjacency.
        edge_index = pd.read_csv(os.path.join(dir, 'edge_list.txt'), header = None).values
        # node features.
        x = pd.read_csv(os.path.join(dir, 'node_attributes.txt'), header = None).values
        # node labels. 
        y = pd.read_csv(os.path.join(dir, 'node_labels.txt'), header = None).values
        lenx = x.shape[0]
        # convert to tensors. 
        edge_index = torch.tensor(edge_index, dtype= torch.long).t().contiguous()
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype= torch.long)
        
        # create a tensor of boolean values 
        # to randomly mask out test data points. 
        train_mask = pd.read_csv(os.path.join(dir, 'train_mask.txt'), header = None).values
        train_mask = torch.tensor(train_mask, dtype = torch.bool).squeeze()

        data = Data(x= x, edge_index = edge_index, y = y, train_mask = train_mask)
        self.data, self.slices = self.collate([data])
    
    def __repr__(self): 
        return '{}()'.format(self.__class__.__name__)

class HARDataCentral(InMemoryDataset): 
    """create a dataset object given a
    directory for fitting a centralized model. 

    create a list instead a single dataframe. 
    """
    def __init__(self, dir, TRAIN_PROP=0.7):
        super().__init__()
        data_components = ['edge_list.txt', 'node_attributes.txt', 'node_labels']
        #TODO: check if all component text files are present. 
        
        datalist = []
        all_agents = os.listdir(dir)
        for each_agent in all_agents: 
            # adjacency.
            edge_index = pd.read_csv(os.path.join(dir, each_agent, 'edge_list.txt'), header = None).values
            # node features.
            # print(edge_index)
            x = pd.read_csv(os.path.join(dir, each_agent, 'node_attributes.txt'), header = None).values
            # node labels. 
            y = pd.read_csv(os.path.join(dir, each_agent, 'node_labels.txt'), header = None).values
    
            # convert to tensors. 
            edge_index = torch.tensor(edge_index, dtype= torch.long).t().contiguous()
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype= torch.long)
            # create a tensor of boolean values 
            # to randomly mask out test data points. 
            train_mask = pd.read_csv(os.path.join(dir, each_agent, 'train_mask.txt'), header = None).values
            train_mask = torch.tensor(train_mask, dtype = torch.bool).squeeze()

            datalist.append(Data(x= x, edge_index = edge_index, y = y, train_mask = train_mask))
        
        self.data, self.slices = self.collate(datalist)
    
    def __repr__(self): 
        return '{}()'.format(self.__class__.__name__)

class GCN_mhealth(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(GCN_mhealth, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier1 = nn.Linear(128, 64)
        self.classifier2 = nn.Linear(64, num_class)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        # Apply a final (linear) classifier.
        h  = self.dropout(h)
        out = self.classifier1(h)
        out = self.dropout(out)
        out = self.classifier2(out)
        return out

class GCN_wisdm(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(GCN_wisdm, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_class)
        

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.dropout(h)
        # Apply a final (linear) classifier.
        h = self.bn1(h)
        h = self.fc2(h)
        h = h.relu()
        h = self.dropout(h)
        out = self.out(h)
        return out 

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.true_divide(w_avg[k], len(w))
    return w_avg

class SimpleHAR(Dataset): 
    def __init__(self, x, y): 
        super().__init__()
        self.x = pd.concat(x)
        self.y = pd.concat(y)
        
    def __len__(self) : 
        return self.x.shape[0]
    
    def __getitem__(self, idx): 
        x = self.x.iloc[idx, :].values
        y = self.y.iloc[idx, :].values

        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        return x, y

class FFN(nn.Module): 
    def __init__(self, input_dim, num_class): 
        super().__init__()

        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x): 
        x = self.dropout(torch.relu(self.layer1(x)))
        x = self.dropout(torch.relu(self.layer2(x)))
        x = self.dropout(torch.relu(self.layer3(x)))
        x = self.out(x)
        return x

