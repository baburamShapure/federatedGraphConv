"""baseline federated model using a feed-forward net 
to compare with centralized gnn. 
"""

import os
import sys 
sys.path.insert(0, os.getcwd())
# print(sys.path)
import pandas as pd 
import numpy as np 
import random
import copy 

from fedgraphconv.prep_mhealth import prep_mhealth
from fedgraphconv.prep_wisdm import prep_wisdm
from fedgraphconv.data_utils import SimpleHAR, HARDataCentral
from fedgraphconv.models import FFN
from fedgraphconv.fed_utils import average_weights

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader


import tqdm 
import argparse

import mlflow 
from mlflow import log_metric, log_param, log_artifacts

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
                    default= 5, 
                    type = int,
                    help = 'Number of local epochs to run')

parser.add_argument('--batch_size', 
                    default= 256, 
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

def evaluate(data): 
    global_model.eval()
    y = data.y.squeeze().t() - 1
    with torch.no_grad():
        out = global_model(data.x)  
        accuracy = torch.mean((torch.argmax(out[~data.train_mask] , 1) == y[~data.train_mask]).float())
    return accuracy


if __name__ == '__main__':

    args = parser.parse_args()
    mlflow.set_experiment('ffn_federated')

    if args.data == 'mhealth': 
        # prep_mhealth(args.num_sample, args.dist_thresh, args.train_prop)
        num_class = 12 
        input_dim = 23
        DATADIR  = 'data/processed/mhealth'
        
    elif args.data == 'wisdm': 
        # prep_wisdm(args.num_sample, args.dist_thresh, args.train_prop)
        num_class = 6
        input_dim = 9
        DATADIR  = 'data/processed/wisdm'

    NUM_ROUNDS = args.num_rounds
    FL_SAMPLE = args.fl_sample
    FL_AGENTS = os.listdir(DATADIR)
    EPOCHS = args.local_epochs
    EPOCHS = 100
    mlflow.log_params({
                    'num_sample': args.num_sample, 
                    'dist_thresh': args.dist_thresh, 
                    'train_prop' : args.train_prop, 
                    'local_epochs' : EPOCHS, 
                    'lr': args.lr,
                    'num_rounds': NUM_ROUNDS,
                    'fl_sample': FL_SAMPLE
                    })

    global_model = FFN(input_dim, num_class) 

    for each_round in tqdm.tqdm(range(NUM_ROUNDS)): 
        agents_to_train = random.sample(FL_AGENTS, k= int(FL_SAMPLE * len(FL_AGENTS)))
        model_list = []
        metrics = {}
        _n = 0
        _a = 0
        for each_agent in agents_to_train: 
            # read the data. 
            x = pd.read_csv(os.path.join(DATADIR, each_agent, 'node_attributes.txt'), 
                header = None)
            y = pd.read_csv(os.path.join(DATADIR, each_agent, 'node_labels.txt'), 
                    header = None)
            trn_msk = pd.read_csv(os.path.join(DATADIR, each_agent, 'train_mask.txt'), 
                    header = None)

            x = [x[trn_msk.values.reshape(-1, 1)]]
            y = [y[trn_msk.values.reshape(-1, 1)]]
                        
            hardata_trn = SimpleHAR(x, y)
            loader_train = DataLoader(hardata_trn, args.batch_size, shuffle = True)
            loss_fn = nn.CrossEntropyLoss()
            model = copy.deepcopy(global_model) 
            optimizer = optim.Adam(model.parameters(), args.lr)#, weight_decay= 0.1)
            for epoch in range(args.local_epochs):
                for i, (x_, y_) in enumerate(loader_train):
                    outs = model(x_)
                    # print(y_.shape, outs.shape)
                    loss = loss_fn(outs, y_.squeeze() - 1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            model_list.append(model.state_dict())
                    
        # average weight at end of round. 
        avg_weights = average_weights(model_list)
        global_model.load_state_dict(avg_weights)
        
        glob_a = 0
        glob_n = 0
        # get accuracy at end of round. 
        dataset  = HARDataCentral(DATADIR, normalize= False)
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
    