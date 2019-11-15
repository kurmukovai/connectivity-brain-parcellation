import numpy as np
from tqdm import tqdm
from sys import argv
from glob import glob
from multilevel import compute_partition, generate_subgraphs, cluster_subgraphs
from load_concon import load_concon

if __name__=="__main__":
    
    sparsity = int(argv[1])
    
    average_desikan = np.load('/data01/ayagoz/sparse_32_concon_HCP/parcellations/desikan_aparc_average_1113.npy',
                   allow_pickle=True)
        
    paths = glob(f'/data01/ayagoz/sparse_32_concon_HCP/connectomes/ConCon_resolution/{sparsity}/*.npz')
    adj = load_concon(paths[0], labels_to_drop=average_desikan)
    
    for file in tqdm(paths[1:]):
        adj_temp = load_concon(file, labels_to_drop=average_desikan)
        adj += adj_temp
       
    partition_level_1 = compute_partition(adj)
    partition_level_2 = cluster_subgraphs(adj, partition_level_1, 5, 200)
    partition_level_3 = cluster_subgraphs(adj, partition_level_2, 5, 360)
    
    parcellation_path = '/data01/ayagoz/sparse_32_concon_HCP/parcellations/ensemble_parcellation/average_network_partition'
    np.save(f'{parcellation_path}/level1/{sparsity}/aver_level1_{sparsity}.npy',
           partition_level_1)
    np.save(f'{parcellation_path}/level2/{sparsity}/aver_level2_{sparsity}.npy',
           partition_level_2)
    np.save(f'{parcellation_path}/level3/{sparsity}/aver_level3_{sparsity}.npy',
           partition_level_3)