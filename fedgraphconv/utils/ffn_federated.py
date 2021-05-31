"""baseline federated model using a feed-forward net 
to compare with centralized gnn. 
"""
import pandas as pd 
import numpy as np 
import os 
import torch 
import torch.nn as nn
import tqdm 
from mlflow import log_metric, log_param, log_artifacts
import mlflow 
from model_utils import SimpleHAR, FFN
from torch.utils.data import DataLoader, Dataset


mlflow.set_experiment('ffn_federated')
datadir = 'data/processed'

agents = os.listdir(datadir)

train_x = []
test_x = []
train_y = []
test_y = []

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

    loader_train = DataLoader(hardata_trn, 256, shuffle = True)
    # loader_tst = DataLoader(hardata_tst, 32, shuffle = True)

    model = FFN() 
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss() 

    EPOCHS = 100
    params = {"epochs": EPOCHS ,
            "lr" : 0.01, 
            "batch_size" :256
            }

    mlflow.log_params(params)

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
    
        x = torch.FloatTensor(test_x[int(each_agent) - 1].values)
        y = torch.LongTensor(test_y[int(each_agent) - 1].values)
        outs = model(x)
        preds = torch.max(outs, 1)[1]
        acc = torch.mean( (preds == y.squeeze() - 1).float()).item()
        metrics['accuracy-agent_{0}'.format(each_agent)]=  acc
        
        mlflow.log_metrics(metrics, step= epoch)
        
