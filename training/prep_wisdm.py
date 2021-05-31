import pandas as pd 
import numpy as np 
import os 
import networkx as nx 
import matplotlib.colors as mcolors
import random 
import scipy.spatial as sp 
import scipy.stats as stats
import tqdm 

# datadir = 'data/WISDM_ar_v1.1'

activity_map={}
activity_map[1]='Walking'
activity_map[2]='Jogging'
activity_map[3]='Upstairs'
activity_map[4]='Downstairs'
activity_map[5]='Sitting'
activity_map[6]='Standing'

activity_map={}
activity_map['Walking']=1
activity_map['Jogging']=2 
activity_map['Upstairs']= 3
activity_map['Downstairs']= 4
activity_map['Sitting']= 5
activity_map['Standing'] = 6
  

def add_encoded_activity(filename, datadir, sep = "\t"):
    """given raw user data 
    add the encoded activity column
    """
    user_data = pd.read_csv(os.path.join(datadir, filename), 
                            sep = sep)
    # print(user_data.shape)
    colnames= ['user_id', 'activity', 'timestamp'] + ['feature_{}'.format(i) for i in range(1, 4)] 
    user_data.columns = colnames
    user_data['encoded_activity'] =  user_data['activity'].map(activity_map)
    # user_data['user_id'] = filename.split('_')[1].split('.')[0][7:]
    user_data = user_data[['user_id', 'encoded_activity', 'feature_1', 'feature_2', 'feature_3']]

    return user_data

def average_slice(df_, NUM_SAMPLE = 128):
    """prepare time slices and 
    average over each time slice. 
    """
    out = []
    num_groups = df_.shape[0] // NUM_SAMPLE
    for i in range(0, df_.shape[0], NUM_SAMPLE): 
        idx = (i , min(df_.shape[0], i + NUM_SAMPLE))    
        tmp = df_.iloc[idx[0]:idx[1], :]
        averaged = pd.DataFrame(tmp.iloc[:, -3:].apply(np.mean)).T
        averaged.columns = [i + '_mean' for i in averaged.columns]
        sd = pd.DataFrame(tmp.iloc[:, -3:].apply(np.std)).T
        sd.columns = [i + '_sd' for i in sd.columns]
        skewness = pd.DataFrame(tmp.iloc[:, -3:].apply(stats.skew)).T
        skewness.columns = [i + '_skew' for i in skewness.columns]
        out.append(pd.concat([averaged, sd, skewness, tmp.iloc[:1, :-3].reset_index(drop = True)], axis = 1))
    out = pd.concat(out)
    out.index = range(out.shape[0])
    return out

def prepare_graph(user_data, THRESHOLD = 3):
    """given the data for a user 
    prepare the graph. 
    """
    # print(user_data.head())
    # prepare the distance matrix. 
    dist_mat = pd.DataFrame(sp.distance.cdist(user_data.iloc[:, :9].values, 
                                               user_data.iloc[:, :9].values, 
                                               metric = 'mahalanobis'))

    cols = random.choices(list(mcolors.CSS4_COLORS.keys()), k =15)
    cols_dict = {}
    for i in range(1, 13):
        cols_dict[i] = cols[i]

    G = nx.Graph() 
    for i, row in user_data.iterrows(): 
        G.add_nodes_from([(i+1, {'features': row[:9]})])
                        
    for idx, row in dist_mat.iterrows(): 
        tmp = row.iloc[idx: ]
        # all elements close to row. First is default by itself. 
        neighbors = list(tmp[tmp <= THRESHOLD].index)

        for each_neighbor in neighbors[1: ]: 
            G.add_edge(idx, each_neighbor, weight = row[each_neighbor])

    return G

def write_node_attributes(G, dir): 
    __  = G.nodes.data()
    with open(os.path.join(dir, 'node_attributes.txt'), 'w') as f: 
        for each_node in __ : 
            if len(each_node) > 0: 
                ftr = each_node[1]['features'].values
                print(ftr)
                for each_line in ftr: 
                    f.writeline(each_line)
                f.writelines('\n')
    f.close()
     
def write_graph(G, dir): 
    """
    write a graph G into a directory dir. 
    """
    with open(os.path.join(dir, 'edge_list.txt'), 'w') as f :
        for line in nx.generate_edgelist(G, delimiter = ',', data = False ):
            f.writelines(line)
            f.writelines('\n')
            f.writelines(','.join(line.split(',')[::-1]))
            f.writelines('\n')
        f.close()

def prep_wisdm(num_sample, dist_thresh, train_prop): 
    print('Preparing Data. ')
    DATADIR = 'data\WISDM'
    for each_file in tqdm.tqdm(os.listdir(DATADIR)):
        if each_file not in ['wisdm_subject'+str(i) for i in [4, 7, 16, 20, 33, 35 ]]:
            # print(each_file)
            user = each_file.split('_')[1].split('.')[0][7:] 
            tmp = add_encoded_activity(each_file, DATADIR, sep =',')
            tmp1 = average_slice(tmp, num_sample)
            
            gr = prepare_graph(tmp1, dist_thresh)

            if user not in os.listdir('data\processed\wisdm'): 
                os.mkdir(os.path.join('data\processed\wisdm', user))
            
            tmp1.iloc[:, :9].to_csv(os.path.join('data\processed\wisdm', user, 'node_attributes' + '.txt'), 
                                    header = None, index = None)
            # prepare training mask. 
            ar = pd.DataFrame(np.random.uniform(0, 1,   
                                tmp1.shape[0]) >= 1 - train_prop, 
                                columns = ['train_mask'])

            tmp1['encoded_activity'].to_csv(os.path.join('data\processed\wisdm', user, 'node_labels' + '.txt'), 
                                            header = None, index = None)
            ar.to_csv(os.path.join('data\processed\wisdm', user, 'train_mask.txt'), 
                                            header = None, index = None)
            write_graph(gr, os.path.join('data\processed\wisdm', user))
    print('Data preparation finished. ')

if __name__ == '__main__': 
    prep_wisdm(128, 1, 0.7)