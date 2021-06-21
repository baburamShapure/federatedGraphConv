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
1. Compare convergence speeds of 4 models in 1 graph. 
2. Print sample graphs from data. 

