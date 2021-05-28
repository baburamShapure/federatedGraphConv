# Federated Graph Convolution Networks and Graph Attention Networks 

How is it like training a federated model on a Graph Neural network?  


## To view our experiments: 

Requires mlflow. Install with 

```bash 
pip install mlflow 
```
and  run 

```bash 
mlflow ui 
```

```batch
./hyper_param_tune_central.bat
```
and 

```batch 
./hyper_param_tune_federated.bat
```

## To run a specific configuration of hyper-params run: 

```bash 
python scripts\gnn_central.py --num_sample 32 --dist_thresh 1
```

## TODO : 

1. aggregate test data for multiple agents in 1 dataset. 
2. report graph fed, graph_central, simple_central & simple_federated in one graph. 


## Experiment Logs: 

### April, 27: 
It is getting tricky to build a good model on the WISDM dataset.  
I tried some combinations of hyper-params, model architecture but no real luck so far. 

I have switched the distance metric from euclidean to mahalanobis to account for variance. 
Results have not improved but the distance metric has become uniform. 

Need to understand bias-variance trade-off in the model. Plot in-sample and outof sample accuracies for comparison. 

### April 28: 

Tried building bigger models on the WISDM dataset. 

There seems to be performance ceiling at 70% accuracy. 
I tracked both in the in-sample and out-of-sample accuracies for all models. 
There is almost no overfitting and bigger models are not helping. 
It makes sense to re-visit the data aggregation process. Perhaps, while creating time buckets, 
too much information has been averaged and models are having a hard time understanding the information. 
Will try to take a look at the aggregation process next. 


### May 19: 

Now I only have 1 graph conv layer. I think the 2 conv layers
were aggregating too much information. 

Results do not seem to be immediately better. 
Will let this run over 1000 epochs and see how they look. 

Next step: Use a scheduler to control the learning rate. 

Need to do a detailed study on the data aggregation process. 


### May 21: 

Remove outlying subjects and re-run model. 
Try a scheduler to control learning rate. 
Open wism.ipynb. 

### May 28
Included sd and skewness for each slice. Overall accuracy is up to 90%. 
Next, switch conv layer with attention. 

