import os 
import sys 
sys.path.insert(0, os.getcwd())

import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
from torch.nn import Linear 
import torch.optim as optim 

import networkx as nx 
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import GCNConv

import argparse
import mlflow 

from fedgraphconv.prep_mhealth import prep_mhealth
from fedgraphconv.prep_wisdm import prep_wisdm

from fedgraphconv.data_utils import HARData, HARDataCentral
from fedgraphconv.models import GCN_mhealth, GCN_mhealth_Attn, GCN_wisdm, GCN_wisdm_Attn

from fedgraphconv.fed_utils import average_weights

import time
import tqdm 
import random
import copy 

parser = argparse.ArgumentParser()

parser.add_argument('--data',
                    default= 'wisdm',
                    help = 'Dataset to use')

parser.add_argument('--num_sample', 
                    default= 32, 
                    type= int,
                    help =  'Number of samples in each window')

parser.add_argument('--dist_thresh', 
                    default= 0.3,
                    type = float, 
                    help =  'Minimum euclidean distance to draw an edge')

parser.add_argument('--train_prop', 
                    default= 0.7, 
                    type = float,
                    help =  'Proportion of data to include in training.')

parser.add_argument('--local_epochs', 
                    default= 10, 
                    type = int,
                    help = 'Number of local epochs to run')

parser.add_argument('--batch_size', 
                    default= 4, 
                    type = int, 
                    help = 'Batch size in each iteration')

parser.add_argument('--lr', 
                    default= 0.01, 
                    type = float, 
                    help = 'Learning rate')

parser.add_argument('--num_rounds', 
                    default= 10, 
                    type = int, 
                    help = 'Number of federated rounds')

parser.add_argument('--fl_sample', 
                    default= 0.4, 
                    type = float, 
                    help = 'Proportion of agents that participate in each federation round')

parser.add_argument('--attention', 
                    default=True,  
                    help = 'Use graph attention instead of convolution.')

def train(data, criterion):
    model.train()
    optimizer.zero_grad()  
    out = model(data.x, data.edge_index)  
    y = data.y.squeeze().t() - 1
    loss = criterion(out[data.train_mask], y[data.train_mask] ) 
    loss.backward() 
    optimizer.step()  
    return loss

def evaluate(data): 
    global_model.eval()
    y = data.y.squeeze().t() - 1
    with torch.no_grad():
        out = global_model(data.x, data.edge_index)  
        accuracy = torch.mean((torch.argmax(out[~data.train_mask] , 1) == y[~data.train_mask]).float())
    return accuracy


if __name__ == '__main__':

    args = parser.parse_args()
    
    if args.attention == True:
        mlflow.set_experiment('gnn_federated_attention')
    else:
        mlflow.set_experiment('gnn_federated')

    DATADIR  = 'data/processed'
    if args.data == 'mhealth': 
        # prep_mhealth(args.num_sample, args.dist_thresh, args.train_prop)
        num_class = 12
        input_dim = 23
        DATADIR  = 'data/processed/mhealth'
        if args.attention:
            global_model = GCN_mhealth_Attn(input_dim, num_class)
        else:
            global_model = GCN_mhealth(input_dim, num_class)
    
    elif args.data == 'wisdm': 
        # prep_wisdm(args.num_sample, args.dist_thresh, args.train_prop)
        num_class = 6
        input_dim = 9
        DATADIR  = 'data/processed/wisdm'
        if args.attention:
            global_model = GCN_wisdm_Attn(input_dim, num_class)
        else:
            global_model = GCN_wisdm(input_dim, num_class)
    
    mlflow.set_tag('dataset', args.data)
    
    FL_AGENTS = os.listdir(DATADIR)
    NUM_ROUNDS = args.num_rounds
    FL_SAMPLE = args.fl_sample
    EPOCHS = args.local_epochs
    
    mlflow.log_params({
                        'num_sample': args.num_sample, 
                        'dist_thresh': args.dist_thresh, 
                        'train_prop' : args.train_prop, 
                        'local_epochs' : EPOCHS, 
                        'lr': args.lr,
                        'num_rounds': NUM_ROUNDS,
                        'fl_sample': FL_SAMPLE
                      })
    
    for each_round in tqdm.tqdm(range(NUM_ROUNDS)): 
        agents_to_train = random.sample(FL_AGENTS, k= int(FL_SAMPLE * len(FL_AGENTS)))
        model_list = []
        metrics = {}
        _n = 0
        _a = 0
        for each_agent in agents_to_train: 
            # read the data. 
            
            dataset  = HARData(os.path.join(DATADIR, str(each_agent)))[0]
            loss = nn.CrossEntropyLoss()
            model = copy.deepcopy(global_model)
            optimizer = optim.Adam(model.parameters(), args.lr, weight_decay= 0.1)
            for epoch in range(EPOCHS):
                loss_ = train(dataset, loss)
            
            model_list.append(model.state_dict())
                    
        # average weight at end of round. 
        avg_weights = average_weights(model_list)
        global_model.load_state_dict(avg_weights)
        
        # get accuracy at end of round. 
        dataset  = HARDataCentral(DATADIR)
        i = 1 
        for each_data in dataset: 
            accuracy = evaluate(each_data) # evaluate the global model on each data. 
            metrics['accuracy-agent_{0}'.format(i)]=  accuracy.item()
            _n += each_data.x[~each_data.train_mask].size()[0]
            _a += each_data.x[~each_data.train_mask].size()[0] * accuracy.item()
            i+=1
        metrics['accuracy'] = _a / _n
        mlflow.log_metrics(metrics, step = each_round)
        # print(metrics['accuracy'])
           