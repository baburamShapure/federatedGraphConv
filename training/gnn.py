import os 
import pandas as pd 
import torch 
import torch.nn as nn 
import numpy as np 
import networkx as nx 
from torch_geometric.data import InMemoryDataset, Data
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
    def __init__(self, dir):
        super().__init__()
        data_components = ['edge_list.txt', 'node_attributes.txt', 'node_labels']
        #TODO: check if all component text files are present. 

        # adjacency.
        edge_index = pd.read_csv(os.path.join(dir, 'edge_list.txt'), header = None).values
        # node features.
        x = pd.read_csv(os.path.join(dir, 'node_attributes.txt'), header = None).values
        # node labels. 
        y = pd.read_csv(os.path.join(dir, 'node_labels.txt'), header = None).values
        
        # convert to tensors. 
        edge_index = torch.tensor(edge_index, dtype= torch.long).t().contiguous()
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype= torch.long)

        data = Data(x= x, edge_index = edge_index, y = y)
        self.data, self.slices = self.collate([data])
    
    def __repr__(self): 
        return '{}()'.format(self.__class__.__name__)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, 12)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        # Apply a final (linear) classifier.
        out = self.classifier(h)
        return out, h

def train(data, criterion):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out, data.y.squeeze().t() - 1)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.true_divide(w_avg[k], len(w))
    return w_avg

# dataset = HARData('data/processed/2')[0]

# # Gather some statistics about the graph. courtesy : Pytorch geometric  example 1. 
# print(f'Number of nodes: {dataset.num_nodes}')
# print(f'Number of edges: {dataset.num_edges}')
# print(f'Average node degree: {dataset.num_edges / dataset.num_nodes:.2f}')
# print(f'Number of training nodes: {dataset.train_mask.sum()}')
# print(f'Training node label rate: {int(dataset.train_mask.sum()) / dataset.num_nodes:.2f}')
# print(f'Contains isolated nodes: {dataset.contains_isolated_nodes()}')
# print(f'Contains self-loops: {dataset.contains_self_loops()}')
# print(f'Is undirected: {dataset.is_undirected()}')

# model = GCN()
# criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.
# for epoch in range(401):
#     loss, h = train(dataset)

DATADIR  = 'data/processed'
FL_AGENTS = [i for i in range(1, 11)]
NUM_ROUNDS = 100
FL_SAMPLE = 0.4
EPOCHS = 100
global_model = GCN()


# LOCAL_MODELS = {}
for each_round in tqdm.tqdm(range(NUM_ROUNDS)): 
    agents_to_train = random.sample(FL_AGENTS, k= int(FL_SAMPLE * len(FL_AGENTS)))
    model_list = []
    for each_agent in agents_to_train: 
        # read the data. 
        dataset  = HARData(os.path.join(DATADIR, str(each_agent)))[0]
        loss = nn.CrossEntropyLoss()
        model = copy.deepcopy(global_model)
        optimizer = optim.Adam(model.parameters())
        model.train()
        for epoch in range(EPOCHS):
            loss_, h = train(dataset, loss)

        # print('Round: {0}, Agent: {1}'.format(each_round, each_agent))
        # print(get_accuracy(model, testLoader))     
        model_list.append(model.state_dict())
    # average weight at end of round. 
    avg_weights = average_weights(model_list)
    global_model.load_state_dict(avg_weights)
    # # test averaged model on each agent. 
    # for each_agent in FL_AGENTS: 
    #   testdata = pd.read_csv(os.path.join(DATADIR, each_agent, 'test.csv'))
    #   testdata = testdata.fillna(0)
    #   test = HARData(testdata)
    #   testloader = DataLoader(test, batch_size = 1024, shuffle= False)
    #   acc = get_accuracy(global_model, testloader)
    #   print("\nRound : {0}, Agent : {1}, Accuracy: {2} \n".format(each_round, each_agent, acc))

