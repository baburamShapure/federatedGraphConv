import pandas as pd 
import numpy as np 
import os 
import networkx as nx 
import matplotlib.colors as mcolors
import random 
import scipy.spatial as sp 


activity_map={}
activity_map[1]='Standing still'
activity_map[2]='Sitting and relaxing'
activity_map[3]='Lying down'
activity_map[4]='Walking'
activity_map[5]='Climbing stairs'
activity_map[6]='Waist bends forward'
activity_map[7]='Frontal elevation of arms'
activity_map[8]='Knees bending'
activity_map[9]='Cycling'
activity_map[10]='Jogging'
activity_map[11]='Running'
activity_map[12]='Jump front & back'
  

def add_encoded_activity(filename, datadir):
    """given raw user data 
    add the encoded activity column
    """
    user_data = pd.read_csv(os.path.join(datadir, filename), sep = '\t', header = None)
    colnames= ['feature_{}'.format(i) for i in range(1, 24)] + ['encoded_activity']
    user_data.columns = colnames
    user_data['activity'] =  user_data['encoded_activity'].map(activity_map)
    user_data['user_id'] = filename.split('_')[1].split('.')[0][7:]
    return user_data[user_data['encoded_activity'] > 0 ]

def average_slice(df_, NUM_SAMPLE = 128):
    """prepare time slices and 
    average over each time slice. 
    """
    out = []
    num_groups = df_.shape[0] // NUM_SAMPLE
    for i in range(0, df_.shape[0], NUM_SAMPLE): 
        idx = (i , min(df_.shape[0], i + NUM_SAMPLE))    
        tmp = df_.iloc[idx[0]:idx[1], :]
        averaged = pd.DataFrame(tmp.iloc[:, :23].apply(np.mean)).T
        out.append(pd.concat([averaged, tmp.iloc[:1, -3:].reset_index(drop = True)], axis = 1))
    out = pd.concat(out)
    out.index = range(out.shape[0])
    return out

def prepare_graph(user_data, THRESHOLD = 3):
    """given the data for a user 
    prepare the graph. 
    """
    # prepare the distance matrix. 
    dist_mat = pd.DataFrame(sp.distance_matrix(user_data.iloc[:, :23].values, 
                                               user_data.iloc[:, :23].values))

    cols = random.choices(list(mcolors.CSS4_COLORS.keys()), k =15)
    cols_dict = {}
    for i in range(1, 13):
        cols_dict[i] = cols[i]

    G = nx.Graph() 
    for i, row in user_data.iterrows(): 
        G.add_nodes_from([(i+1, {'features': row[:23], 
                              'label': row['encoded_activity'], 
                              'color': cols[row['encoded_activity']]})])
    
    for idx, row in dist_mat.iterrows(): 
        tmp = row.iloc[idx: ]
        # all elements close to row. First is default by itself. 
        neighbors = list(tmp[tmp <= THRESHOLD].index)

        for each_neighbor in neighbors[1: ]: 
            G.add_edge(idx,  each_neighbor , weight = row[each_neighbor])

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
