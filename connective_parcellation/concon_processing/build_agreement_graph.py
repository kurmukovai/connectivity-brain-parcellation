import numpy as np
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from sys import argv

def n_agreements(v1, v2, partitions):
    '''
    Compute number of agreements for two nodes
    ---
    
    Parameters
    ---
    
    v1 - int,
     node 1 index
     
    v2 - int,
     node 2 index
     
    partitions - array,
     n_partitions x n_nodes array, every row is a single graph partition labels
     
    Returns
    ---
    
    percentage of agreements between 2 node coloring in all partitions
    '''  
    n_partitions = partitions.shape[0]
    subset = partitions[:, [v1, v2]]
    n_agr = np.where(subset[:, 0]==subset[:, 1])[0].shape[0]
    return n_agr / n_partitions


def compute_agreements(partitions): 
    '''
    Construct agreement graph from set of partitions, see CSPA
    ---
    
    Parameters
    ---
     
    partitions - array,
     n_partitions x n_nodes array, every row is a single graph partition labels
     
    Returns
    ---
    
    adjacency matrix of agreement graph
    '''
    
    n_objects = partitions.shape[1] # for ConCon n_objects = 20484
    _pairs = np.triu_indices(n_objects, k = 1)
    pairs = np.zeros((_pairs[0].shape[0], 2), dtype=int)
    pairs[:, 0], pairs[:, 1] = _pairs[0], _pairs[1]
    adj = np.zeros((n_objects, n_objects))
    
    for pair in tqdm(pairs):
        v1, v2 = pair[0], pair[1]
        adj[v1, v2] = n_agreements(v1, v2, partitions)
        adj[v2, v1] = adj[v1, v2]      

    return adj

def do_job(level, sparsity):
    p = '/data01/ayagoz/sparse_32_concon_HCP/'
    files = glob(f'{p}parcellations/connectivity_parcellation_level{level}/{sparsity}/*.npy')
    partitions = []
    for file in files:
        labels = np.load(file)
        partitions.append(labels)
    partitions = np.array(partitions)
    adj = compute_agreements(partitions)
    np.save(f'{p}connectomes/agreement_graphs/level{level}/agreement_adj_{level}_{sparsity}.npy', adj)

if __name__ == '__main__':
    level = argv[1] # 1, 2, 3
    Parallel(n_jobs=10)(delayed(do_job)(level, sparsity) for sparsity in tqdm(range(10, 101, 10)))