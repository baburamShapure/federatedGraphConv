import pandas as pd 
import os 
from .utils import add_encoded_activity, average_slice, prepare_graph, write_graph
import networkx as nx
import numpy as np 
import argparse
import shutil 

parser = argparse.ArgumentParser()

def prep_mhealth(num_sample,  dist_thresh, train_prop):
    # change to others. 
    print('Preparing Data. ')
    DATADIR = 'data\MHEALTHDATASET'
    # # check if data exists in cache. 
    foldertolookfor = 'mhealth_{0}_{1}_{2}'.format(num_sample, dist_thresh, train_prop)
    if foldertolookfor in os.listdir(os.path.join('data', 'processed', 'cached')):
        print('Found in cache.\n')
        # data already exists. Just copy. 
        
        shutil.rmtree(os.path.join('data', 'processed', 'mhealth'))
        
        shutil.copytree(os.path.join('data', 'processed', 'cached', foldertolookfor), 
                        os.path.join('data', 'processed', 'mhealth'))
    
    else:
        print("Did not find data in Cache. ")
        cache_dir = os.path.join('data', 'processed', 'cached', foldertolookfor)           
        print(cache_dir)
        os.mkdir(cache_dir)
        #TODO: parallelize this
        for each_file in os.listdir(DATADIR):
            if 'log' in each_file: 
                user = each_file.split('_')[1].split('.')[0][7:] 
                tmp = add_encoded_activity(each_file, DATADIR)
                tmp1 = average_slice(tmp, num_sample)
                gr = prepare_graph(tmp1, dist_thresh)
                
                if user not in os.listdir('data\processed\mhealth'): 
                    os.mkdir(os.path.join('data\processed\mhealth', user))
                
                if user not in os.listdir(cache_dir): 
                    os.mkdir(os.path.join(cache_dir, user))
                
                tmp1.iloc[:, :23].to_csv(os.path.join('data\processed', user, 'node_attributes' + '.txt'), 
                                        header = None, index = None)
                # prepare training mask. 
                ar = pd.DataFrame(np.random.uniform(0, 1,   
                                tmp1.shape[0]) >= 1 - train_prop, 
                                columns = ['train_mask'])

                tmp1['encoded_activity'].to_csv(os.path.join('data\processed', user, 'node_labels' + '.txt'), 
                                                header = None, index = None)
                ar.to_csv(os.path.join('data\processed', user, 'train_mask.txt'), 
                                                header = None, index = None)
                write_graph(gr, os.path.join('data\processed', user))

                # write in cache dir also. 
                tmp1.iloc[:, :23].to_csv(os.path.join(cache_dir, user, 'node_attributes' + '.txt'), 
                                        header = None, index = None)
                tmp1['encoded_activity'].to_csv(os.path.join(cache_dir, user, 'node_labels' + '.txt'), 
                                                header = None, index = None)
                ar.to_csv(os.path.join(cache_dir, user, 'train_mask.txt'), 
                                                header = None, index = None)
                write_graph(gr, os.path.join(cache_dir, user))


parser.add_argument('--num_sample', 
                    type = int,
                    default= 128, 
                    help =  'Number of samples in each window')

parser.add_argument('--dist_thresh', 
                    default= 0.3, 
                    type = float,
                    help =  'Minimum euclidean distance to draw an edge')

parser.add_argument('--train_prop', 
                    default= 0.7, 
                    type = float,
                    help =  'Proportion of data to include in training.')


if __name__ == '__main__':
    args = parser.parse_args()
    prep_mhealth(args.num_sample, args.dist_thresh, args.train_prop)
