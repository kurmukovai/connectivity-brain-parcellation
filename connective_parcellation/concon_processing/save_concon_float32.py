'''
Saves all concons in float32 format
'''

import numpy as np    
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm

def load(path):
    data = np.load(path)
    return data

def save(path, data, arr):
    from scipy.sparse import coo_matrix, save_npz
    sparse_data = coo_matrix((arr, (data['row'], data['col'])))
    save_npz(path, sparse_data)

def load_save(path):
    data = load(path)
    arr = data['data'].astype(np.float32)
    new_path = f'/data01/ayagoz/sparse_32_concon_HCP/{path.split("/")[-1]}'
    save(new_path, data, arr)

Parallel(n_jobs=15)(delayed(load_save)(file) for file in tqdm(glob('/data01/ayagoz/sparse_old_concon_HCP/*.npz')))