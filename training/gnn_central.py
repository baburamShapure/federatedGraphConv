"""fit a centralized model 
for comparing with federated learning. 
"""

import os
import sys 
sys.path.insert(0, os.getcwd())
# print(sys.path)
import pandas as pd 
import numpy as np 

from fedgraphconv.prep_mhealth import prep_mhealth
from fedgraphconv.prep_wisdm import prep_wisdm
from fedgraphconv.data_utils import HARDataCentral
from fedgraphconv.models import GCN_mhealth, GCN_wisdm, GCN_wisdm_2conv, GCN_wisdm_3conv

import torch 
import torch.nn as nn 
import torch.optim as optim 
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

import tqdm 
import argparse

import mlflow 
from mlflow import log_metric, log_param, log_artifacts

import datetime as dttm 
since = dttm.datetime.now()
since_str = dttm.datetime.strftime(since, '%d-%m-%y %H:%M:%S')


def train(data, criterion):
    model.train()
    optimizer.zero_grad()  
    out = model(data.x, data.edge_index)
    y = data.y.squeeze().t() - 1
    loss = criterion(out[data.train_mask], y[data.train_mask]  ) 
    accuracy = torch.mean((torch.argmax(out[~data.train_mask] , 1) == y[~data.train_mask]).float())
    loss.backward() 
    optimizer.step()
    # scheduler.step()  
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

    mlflow.set_experiment('gnn_central_1')

    args = parser.parse_args()    
    if args.data == 'mhealth': 
        prep_mhealth(args.num_sample, args.dist_thresh, args.train_prop)
        num_class = 12 
        input_dim = 23
        DATADIR  = 'data/processed/mhealth'
        model = GCN_mhealth(input_dim, num_class)
        
    
    elif args.data == 'wisdm': 
        prep_wisdm(args.num_sample, args.dist_thresh, args.train_prop)
        num_class = 6
        input_dim = 9
        DATADIR  = 'data/processed/wisdm'
        model = GCN_wisdm(input_dim, num_class)

    model_params = sum(p.numel() for p in model.parameters())

    mlflow.set_tag('dataset', args.data)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    print('datadir: {0}'.format(DATADIR))
    dataset  = HARDataCentral(DATADIR)
    loader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle = True)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    mlflow.log_params({'num_sample': args.num_sample, 
                      'dist_thresh': args.dist_thresh, 
                      'train_prop' : args.train_prop, 
                      'batch_size': BATCH_SIZE, 
                      'max_epochs' : EPOCHS, 
                      'lr': args.lr,
                      'num_parameters': model_params
                      })

    logs = pd.DataFrame(columns=['epoch', 'agent', 'accuracy'])

    excel = []
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
        now = dttm.datetime.now()
        excel.append((epoch, since_str, glob_a/glob_n, now.strftime('%y-%m-%d %H:%M:%S'), (now-since).total_seconds()))

    df = pd.DataFrame(excel)
    df.columns =['epoch', 'time_start', 'accuracy', 'log_time', 'time_elapsed']
    df.to_csv('logs_{0}_gnn_central.csv'.format(args.data), index= None)