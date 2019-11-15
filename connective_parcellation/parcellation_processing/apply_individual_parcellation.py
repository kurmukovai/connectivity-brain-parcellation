import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from load_concon import load_concon
from sys import argv

def squeeze_matrix(matrix,
                   labels,
                   drop_minus_1 = False,
                   fill_diag_zero=True,
                   return_transform=False,):
    '''
    Apply parcellation to adjacency matrix
    
    Parameters
    ----
    
    matrix - ndarray,
     graph adjacency matrix M x M
     
    labels - ndarray,
     parcellation labels 1 x M with N unique values
     (cluster/parcels labels)
     
    drop_minus_1 - bool,
     default True, drops rows/columns which
     corresponds to label "-1"
    
    fill_diag_zero - bool,
     default, True, fills diagonal of the
     returned adjacency matrix with zeros    
              
    return_transform - bool,
     default False
    '''
    if drop_minus_1:
        mask = labels != -1
        labels = labels[mask]
        matrix = matrix[mask, :][:, mask]

    input_size = matrix.shape[0]
    output_size = np.unique(labels).shape[0]
    
    d = dict(zip(np.unique(labels), np.arange(output_size)))
    _labels = list(map(lambda x: d[x], labels))
    transform = np.zeros((input_size, output_size))
    
    for row, col in enumerate(_labels):
        transform[row, col] = 1
    
    squeezed = transform.T.dot(matrix.dot(transform))
    squeezed = (squeezed + squeezed.T)/2
    
    if fill_diag_zero:
        np.fill_diagonal(squeezed, 0)
    
    if return_transform:
        return np.array(squeezed), transform
    
    return np.array(squeezed)

def apply_parcellation(subject):
    '''
    Applies parcellation for all sparsity levels
    
    Parameters
    ------
    
    subject - int,
     subject id
    '''
    average_desikan = np.load('/data01/ayagoz/sparse_32_concon_HCP/parcellations/desikan_aparc_average_1113.npy',
                   allow_pickle=True)
    source = '/data01/ayagoz/sparse_32_concon_HCP/connectomes/ConCon_resolution'
    parcellation = '/data01/ayagoz/sparse_32_concon_HCP/parcellations/connectivity_parcellation_level'
    target_folder = '/data01/ayagoz/sparse_32_concon_HCP/connectomes/individual_louvain_parcellation'

    for sparsity in range(10, 101, 10):
        try:
            adj = load_concon(f'{source}/{sparsity}/{subject}.npz', labels_to_drop=average_desikan)
#             print(f'{source}/{sparsity}/{subject}.npz')
            for i in [1,2,3]:
                labels_parcellation = np.load(f'{parcellation}{i}/{sparsity}/{subject}.npy')
#                 print(f'{parcellation}{i}/{sparsity}/ensemble_{i}_{sparsity}.npy')
                adj_parcellation = squeeze_matrix(adj, labels_parcellation)
                np.save(f'{target_folder}/level{i}/{sparsity}/{subject}.npy', adj_parcellation)
#                 print(f'{target_folder}/level{i}/{sparsity}/{subject}.npy')
        except BaseException as e:
            print(e, subject, sparsity, plevel)


if __name__ == "__main__":
    start = int(argv[1])
    end = int(argv[2])
    source = '/data01/ayagoz/sparse_32_concon_HCP/connectomes/ConCon_resolution'
    all_subjects = [uid.split('.')[0] for uid in os.listdir(f'{source}/10')]
    Parallel(n_jobs=25)(delayed(apply_parcellation)(subject) for subject in tqdm(all_subjects[start: end]))
    
    