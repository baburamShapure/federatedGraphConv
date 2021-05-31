"""fit a centralized model 
for comparing with federated learning. 
"""
import os 
import pandas as pd 
import torch 
import torch.nn as nn 
import numpy as np 
import networkx as nx 
from torch_geometric.data import InMemoryDataset, Data
from prep_mhealth import prep_mhealth
from prep_wisdm import prep_wisdm
from torch.nn import Linear 
import torch.optim as optim 
from torch_geometric.nn import GCNConv, GATConv
import time
import tqdm 
import random
import copy
from torch_geometric.data import DataLoader
from model_utils import * 
import datetime as dttm 
import argparse
from mlflow import log_metric, log_param, log_artifacts
import mlflow 

def train(data, criterion):
    model.train()
    optimizer.zero_grad()  
    out = model(data.x, data.edge_index)
    y = data.y.squeeze().t() - 1
    loss = criterion(out[data.train_mask], y[data.train_mask]  ) 
    accuracy = torch.mean((torch.argmax(out[~data.train_mask] , 1) == y[~data.train_mask]).float())
    loss.backward() 
    optimizer.step()  
    return loss

def evaluate(data, test= True): 
        model.eval()
        y = data.y.squeeze().t() - 1
        out = model(data.x, data.edge_index)  

        if test: 
            accuracy = torch.mean((torch.argmax(out[~data.train_mask] , 1) == y[~data.train_mask]).float())
        else: 
            accuracy = torch.mean((torch.argmax(out[data.train_mask] , 1) == y[data.train_mask]).float())
        
        return accuracy

parser = argparse.ArgumentParser()

parser.add_argument('--data',
                    help = 'Dataset to use',
                    default='wisdm')

parser.add_argument('--num_sample', 
                    default= 32, 
                    type= int,
                    help =  'Number of samples in each window')

parser.add_argument('--dist_thresh', 
                    default= 1,
                    type = float, 
                    help =  'Minimum euclidean distance to draw an edge')

parser.add_argument('--train_prop', 
                    default= 0.7, 
                    type = float,
                    help =  'Proportion of data to include in training.')

parser.add_argument('--epochs', 
                    default= 100, 
                    type = int,
                    help = 'Number of epochs to run')

parser.add_argument('--batch_size', 
                    default= 4, 
                    type = int, 
                    help = 'Batch size in each iteration')

parser.add_argument('--lr', 
                    default= 0.01, 
                    type = float, 
                    help = 'Learning rate')

if __name__ == '__main__':

    mlflow.set_experiment('gnn_central_attention')
    
    args = parser.parse_args()    
    if args.data == 'mhealth': 
        # prep_mhealth(args.num_sample, args.dist_thresh, args.train_prop)
        num_class = 12
        input_dim = 23
        DATADIR  = 'data/processed/mhealth'
        model = GCN_mhealth(input_dim, num_class)
    
    elif args.data == 'wisdm': 
        prep_wisdm(args.num_sample, args.dist_thresh, args.train_prop)
        num_class = 6
        input_dim = 9
        DATADIR  = 'data/processed/wisdm'
        model = GCN_wisdm_Attn(input_dim, num_class)

    mlflow.set_tag('dataset', args.data)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    
    print('datadir: {0}'.format(DATADIR))
    dataset  = HARDataCentral(DATADIR)
    loader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle = True)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    mlflow.log_params({'num_sample': args.num_sample, 
                      'dist_thresh': args.dist_thresh, 
                      'train_prop' : args.train_prop, 
                      'batch_size': BATCH_SIZE, 
                      'max_epochs' : EPOCHS, 
                      'lr': args.lr
                      })

    logs = pd.DataFrame(columns=['epoch', 'agent', 'accuracy'])

    for epoch in tqdm.tqdm(range(EPOCHS)):
        for i, batch in enumerate(loader): 
            try:
                loss_ = train(batch, loss)     
            except:
                
                pass 
        step = 0 
        ii=1
        metrics = {}
        glob_a = 0
        glob_n = 0
        for each_data in dataset: 
            try:
                accuracy = evaluate(each_data)
            except Exception as e : 
                print('i :->', ii )
                print(e)
            glob_n += each_data.num_nodes
            glob_a += each_data.num_nodes * accuracy.item()

            metrics['accuracy-agent_{0}'.format(ii)]=  accuracy.item()
            ii+=1 
        metrics['accuracy'] = glob_a / glob_n
        mlflow.log_metrics(metrics, step = epoch)
        step += 1
         