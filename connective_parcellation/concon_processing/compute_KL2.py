import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
from glob import glob
from sys import argv
from joblib import Parallel, delayed
from tqdm import tqdm

def squeeze_matrix(matrix, labels, return_transform=True, fill_diag_zero=True):

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



def KL_distance(CC_adjacency, parcellation):
  
    n_vertices = CC_adjacency.shape[0]
    # squeeze_matrix return downsampled ConCon matrix according to parcellation
    # and apropriate transformation matrix of size n_ConCon_vertices x n_downsample_vertices
    adj, transform = squeeze_matrix(CC_adjacency, parcellation, return_transform=True, fill_diag_zero=False)
    
    # restore ConCon resolution matrix. mean(x)
    t_w = transform / transform.sum(axis=0) # in order to get mean 
    CC_adj_mean = t_w.dot(adj).dot(t_w.T)
    
    i,j = np.triu_indices(n_vertices, k=1)

    ##
    # compute x_i - mean(x)
    ##
    triu_adj = np.array(CC_adjacency[i,j]).reshape(-1)
    triu_adj_mean = np.array(CC_adj_mean[i,j]).reshape(-1)
    non_zero = np.nonzero(triu_adj)[0]
    triu_adj = triu_adj[non_zero]
    triu_adj_mean = triu_adj_mean[non_zero]
    
    
    return (np.multiply(triu_adj, np.log(triu_adj/triu_adj_mean))).mean()#, non_zero


def load_concon(path, labels_to_drop=None):
    '''
    Load ConCon ajancency matrix
    
    Parameters
    -------
    
    path - str,
     path to concon *.npz file
     
    labels_to_drop - ndarray,
     default None, drops rows/columns which
     corresponds to label "-1" (corpus collosum, cerebellum)
     
    Return
    -------
    
    adjacency matrix
    '''

    data = np.load(path)
    sparse_data = coo_matrix((data['data'], (data['row'], data['col'])))
    adj = sparse_data.todense()
    adjacency_balanced = (adj + adj.T) / 2
    np.fill_diagonal(adjacency_balanced, 0)

    if labels_to_drop is not None:
        mask = labels_to_drop != -1
        adjacency_balanced = adjacency_balanced[mask, :][:, mask]
    return adjacency_balanced

def computeKL(subject_path, parcellations=None, desik=None):
    cc = load_concon(subject_path, labels_to_drop=desik)
    kl = {}
    for key, parcellation in tqdm(parcellations.items()):
        if 'none' in key:
            parcellation = parcellation[desik!=-1]
        kl[key] = [KL_distance(cc, parcellation)]
    kl = pd.DataFrame(kl)
    kl.to_csv(f"/home/kurmukov/connective_parcellation/concon_processing/kl_results/{subject_path.split('/')[-1].split('.')[0]}.csv")

if __name__ == '__main__':
    start = int(argv[1])
    end = int(argv[2])
    df_parcellations = pd.read_csv('/home/kurmukov/connective_parcellation/all_parcellation_paths.csv', index_col=0).query('sparsity=="10" or sparsity == "none"')
    desik = np.load('/data01/ayagoz/sparse_32_concon_HCP/parcellations/desikan_aparc_average_1113.npy')
    cc_paths = glob('/data01/ayagoz/sparse_32_concon_HCP/connectomes/ConCon_resolution/10/*.npz')
    parcellations = {}
    for row in tqdm(df_parcellations.itertuples()):
        parcellations[row.key] = np.load(row.path_to_partition)
    Parallel(20)(delayed(computeKL)(cc, parcellations, desik) for cc in tqdm(cc_paths[start: end]))
