import numpy as np
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from sys import argv
from multilevel import compute_partition

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


# def compute_agreements(partitions, n_jobs=25): 
#     '''
#     Construct agreement graph from set of partitions, see CSPA
#     ---
    
#     Parameters
#     ---
     
#     partitions - array,
#      n_partitions x n_nodes array, every row is a single graph partition labels
     
#     Returns
#     ---
    
#     adjacency matrix of agreement graph
#     '''
    
#     n_objects = partitions.shape[1] # for ConCon n_objects = 20484
#     row, col = np.triu_indices(n_objects, k = 1)
#     adj = np.zeros((n_objects, n_objects))
#     from joblib import Parallel, delayed
#     res = Parallel(n_jobs=n_jobs)(delayed(n_agreements)(v1, v2, partitions) for v1, v2 in tqdm(zip(row, col)))
#     adj[row, col] = res
#     adj[col, row] = res

#     return adj

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
    row, col = np.triu_indices(n_objects, k = 1)
    adj = np.zeros((n_objects, n_objects))
    
    for v1, v2 in tqdm(zip(row, col)):
        adj[v1, v2] = n_agreements(v1, v2, partitions)
        adj[v2, v1] = adj[v1, v2]      

    return adj

def do_job(r=None):
    level = 3
    sparsity = 10
    p = '/data01/ayagoz/sparse_32_concon_HCP/'
    random = np.random.RandomState(r)
    files = glob(f'{p}parcellations/connectivity_parcellation_level{level}/{sparsity}/*.npy')
    random.shuffle(files)
    partitions = []
    for file in files[:50]:
        labels = np.load(file)
        partitions.append(labels)
    adj = compute_agreements(np.array(partitions))
    labels = compute_partition(adj)
    np.save(f'/home/kurmukov/subject_stability/CSPA/CSPA_3_10_{r}.npy', labels)

if __name__ == '__main__':
    start = int(argv[1])
    res = Parallel(n_jobs=10)(delayed(do_job)(r) for r in tqdm(range(start, start+20)))