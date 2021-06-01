import os 
import pandas as pd 

import numpy as np 
import torch 
from torch.utils.data import  Dataset
from torch_geometric.data import InMemoryDataset, Data

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

