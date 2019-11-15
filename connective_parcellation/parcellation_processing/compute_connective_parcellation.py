import os
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
from sys import argv
from joblib import Parallel, delayed
from igraph import Graph
from igraph import ADJ_MAX
from multilevel import compute_partition, generate_subgraphs, cluster_subgraphs
from load_concon import load_concon


def get_partitions_at_levels(subject_id, labels_to_drop):
    
    for sparsity in tqdm(range(10, 101, 10)):
        try:
            path = f'/data01/ayagoz/sparse_32_concon_HCP/ConCon_resolution/{sparsity}/{subject_id}.npz'
            adj = load_concon(path, labels_to_drop=labels_to_drop)
            partition_level_1 = compute_partition(adj)
            partition_level_2 = cluster_subgraphs(adj, partition_level_1, 5, 200)
            partition_level_3 = cluster_subgraphs(adj, partition_level_2, 5, 360)
            parcellation_path = '/data01/ayagoz/sparse_32_concon_HCP/parcellations'
            np.save(f'{parcellation_path}/connectivity_parcellation_level1/{sparsity}/{subject_id}.npy',
                   partition_level_1)
            np.save(f'{parcellation_path}/connectivity_parcellation_level2/{sparsity}/{subject_id}.npy',
                   partition_level_2)
            np.save(f'{parcellation_path}/connectivity_parcellation_level3/{sparsity}/{subject_id}.npy',
                   partition_level_3)
        except BaseException as e:
            print(e, subject_id)
            


if __name__=="__main__":
    
    start = int(argv[1])
    end = int(argv[2])
    
    average_desikan = np.load('/data01/ayagoz/sparse_32_concon_HCP/parcellations/desikan_aparc_average_1113.npy',
                   allow_pickle=True)
    
#     all_subjects = [uid.split('.')[0] for uid in os.listdir('/data01/ayagoz/sparse_32_concon_HCP/ConCon_resolution/10')]
#     Parallel(n_jobs=15)(delayed(get_partitions_at_levels)(subject,
#                                                           average_desikan) for subject in tqdm(all_subjects[start:end]))
    
    
    all_subjects = ['201111', '186141', '162329', '322224', '138231', '154431', '159441', '192843', '136833', '156536', '311320', '166438', '187345', '212419', '268749', '118023', '310621', '154835', '167036', '130821', '198451', '135528', '200210', '280739', '223929', '107018', '188448', '127630', '285345', '202719', '150726', '179548', '246133', '173738', '169444', '268850', '182739', '107725', '164939', '201414', '117930', '105014', '209127', '161327', '140925', '165638', '199453', '112314', '165032', '123420']
#    for subject in tqdm(all_subjects[start:end]):
#        get_partitions_at_levels(subject, average_desikan)
        
    Parallel(n_jobs=10)(delayed(get_partitions_at_levels)(subject,
                                                           average_desikan) for subject in tqdm(all_subjects[start:end]))
    
