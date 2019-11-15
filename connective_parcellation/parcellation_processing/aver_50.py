import numpy as np
from tqdm import tqdm
from sys import argv
from glob import glob
from multilevel import compute_partition, generate_subgraphs, cluster_subgraphs
from load_concon import load_concon

if __name__=="__main__":
    
    random_state = int(argv[1])
    for r in tqdm(range(random_state, random_state+10)):
        random = np.random.RandomState(random_state)
        sparsity = 10
        average_desikan = np.load('/data01/ayagoz/sparse_32_concon_HCP/parcellations/desikan_aparc_average_1113.npy',
                       allow_pickle=True)

        paths = glob(f'/data01/ayagoz/sparse_32_concon_HCP/connectomes/ConCon_resolution/{sparsity}/*.npz')
        random.shuffle(paths)
        adj = load_concon(paths[0], labels_to_drop=average_desikan)

        for file in tqdm(paths[1:50]):
            adj_temp = load_concon(file, labels_to_drop=average_desikan)
            adj += adj_temp

        partition_level_1 = compute_partition(adj)
        partition_level_2 = cluster_subgraphs(adj, partition_level_1, 5, 200)
        partition_level_3 = cluster_subgraphs(adj, partition_level_2, 5, 360)

        np.save(f'/home/kurmukov/subject_stability/aver_50_level1_{sparsity}_{r}.npy',
               partition_level_1)
        np.save(f'/home/kurmukov/subject_stability/aver_50_level2_{sparsity}_{r}.npy',
               partition_level_2)
        np.save(f'/home/kurmukov/subject_stability/aver_50_level3_{sparsity}_{r}.npy',
               partition_level_3)
