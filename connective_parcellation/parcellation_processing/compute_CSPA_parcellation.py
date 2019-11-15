import os
import numpy as np
from scipy.sparse import coo_matrix
# from joblib import Parallel, delayed
from multilevel import compute_partition
from load_concon import load_concon
from tqdm import tqdm
# from sys import argv

def get_partitions(sparsity, level):
    try:
        path = f'/data01/ayagoz/sparse_32_concon_HCP/connectomes/agreement_graphs/level{level}/agreement_adj_{level}_{sparsity}.npy'
        adj = np.load(path)
        partition = compute_partition(adj)
        parcellation_path = f'/data01/ayagoz/sparse_32_concon_HCP/parcellations/ensemble_parcellation/CSPA/level{level}/CSPA_{level}_{sparsity}.npy'
        np.save(parcellation_path, partition)
    except BaseException as e:
        print(e)

if __name__=="__main__":
#     sparsity = argv[1]
    level = [1] * 10 + [2] * 10 + [3] * 10
    sparsity = list(range(100, 9, -10)) * 3
#     Parallel(n_jobs=3)(delayed(get_partitions)(sparsity, level) for level in [1,2,3])
       
    for l, s in tqdm(zip(level, sparsity)):
        get_partitions(s, l)