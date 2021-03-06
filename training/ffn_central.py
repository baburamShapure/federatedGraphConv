"""baseline central model using a feed-forward net 
to compare with centralized gnn. 
"""
import os
import sys 
sys.path.insert(0, os.getcwd())
# print(sys.path)
import pandas as pd 
import numpy as np 

from fedgraphconv.prep_mhealth import *
from fedgraphconv.prep_wisdm import *

from fedgraphconv.data_utils import SimpleHAR
from fedgraphconv.models import FFN 

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

parser = argparse.ArgumentParser()

parser.add_argument('--data',
                    help = 'Dataset to use',
                    default='mhealth')

parser.add_argument('--num_sample', 
                    default= 128, 
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
                    default= 128, 
                    type = int, 
                    help = 'Batch size in each iteration')

parser.add_argument('--lr', 
                    default= 0.01, 
                    type = float, 
                    help = 'Learning rate')

args = parser.parse_args() 

mlflow.set_experiment('ffn_central_1')

if args.data == 'mhealth': 
    prep_mhealth(args.num_sample, args.dist_thresh, args.train_prop)
    num_class = 12
    input_dim = 23
    DATADIR  = 'data/processed/mhealth'
    model = FFN(input_dim, num_class)

elif args.data == 'wisdm': 
    prep_wisdm(args.num_sample, args.dist_thresh, args.train_prop)
    num_class = 6
    input_dim = 9
    DATADIR  = 'data/processed/wisdm'
    model = FFN(input_dim, num_class)

agents = [i for i in os.listdir(DATADIR) if i != 'later']
# print(agents)
mlflow.set_tag('dataset', args.data)


train_x = []
test_x = []
train_y = []
test_y = []

datadir = DATADIR #weird!

for each_agent in agents: 
    x = pd.read_csv(os.path.join(datadir, each_agent, 'node_attributes.txt'), 
                header = None)
    
    y = pd.read_csv(os.path.join(datadir, each_agent, 'node_labels.txt'), 
                    header = None)
    
    trn_msk = pd.read_csv(os.path.join(datadir, each_agent, 'train_mask.txt'), 
                    header = None)

    train_x.append(x[trn_msk.values.reshape(-1, 1)])
    test_x.append(x[~trn_msk.values.reshape(-1, 1)])
    train_y.append(y[trn_msk.values.reshape(-1, 1)])
    test_y.append(y[~trn_msk.values.reshape(-1, 1)])

hardata_trn = SimpleHAR(train_x, train_y)

loader_train = DataLoader(hardata_trn, args.batch_size, shuffle = True)
# loader_tst = DataLoader(hardata_tst, 32, shuffle = True)




# model = FFN() 
print('Number of trainable parameters: ', sum(p.numel() for p in model.parameters()))

optimizer = optim.Adam(model.parameters(), lr = args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
loss_fn = nn.CrossEntropyLoss() 

EPOCHS = args.epochs
params = {"epochs": EPOCHS ,
          "lr" : args.lr, 
          "batch_size" :args.batch_size,
          'num_sample': args.num_sample,
          'dist_thresh': args.dist_thresh
         }

mlflow.log_params(params)
excel = []

for epoch in tqdm.tqdm(range(EPOCHS)):
    model.train()
    for i, (x, y) in enumerate(loader_train):
        outs = model(x)
        loss = loss_fn(outs, y.squeeze() - 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    metrics = {} 
    model.eval()
    
    glob_a = 0
    glob_n = 0
    for each_agent in agents: 
        x = torch.FloatTensor(test_x[int(each_agent) - 1].values)
        y = torch.LongTensor(test_y[int(each_agent) - 1].values)
        outs = model(x)
        preds = torch.max(outs, 1)[1]
        acc = torch.mean( (preds == y.squeeze() - 1).float()).item()
        metrics['accuracy-agent_{0}'.format(each_agent)]=  acc
        glob_a += acc * preds.size()[0]
        glob_n += preds.size()[0]
     
    metrics['accuracy'] = glob_a / glob_n    
    mlflow.log_metrics(metrics, step= epoch)
    now = dttm.datetime.now()
    excel.append((epoch, since_str, glob_a/glob_n, now.strftime('%y-%m-%d %H:%M:%S'), (now-since).total_seconds()))
    
df = pd.DataFrame(excel)
df.columns =['epoch', 'time_start', 'accuracy', 'log_time', 'time_elapsed']
df.to_csv('logs_{0}_ffn_central.csv'.format(args.data), index= None)