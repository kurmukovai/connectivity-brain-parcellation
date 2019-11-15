import numpy as np
from scipy.sparse import coo_matrix

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