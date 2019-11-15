'''
Create sparser concons: 10% edges, ... 90% edges
Sparser concons saved to folders: 90% edges to folder /90, 80% edges to folder /80, etc.
'''
import sys
import numpy as np    
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm


def save(path, arr, row, col):
    from scipy.sparse import coo_matrix, save_npz
    sparse_data = coo_matrix((arr, (row, col)), dtype=np.float32, shape=(20484, 20484))
    save_npz(path, sparse_data)

def sparsify(path):
    data = np.load(path)
    arr = data['data'].astype(np.float32)
    sparsity = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    quantiles = np.quantile(arr, sparsity/100)
    for q, sparsity in zip(quantiles, sparsity):
        new_path = f'/data01/ayagoz/sparse_32_concon_HCP/{100-sparsity}/{path.split("/")[-1]}'
        mask = arr > q
        sparsified_arr = arr[mask]
        row = data['row'][mask] # max index is 20_483, doesnt actually save as 16
        col = data['col'][mask]
        save(new_path, sparsified_arr, row, col)

# file = glob('/data01/ayagoz/sparse_32_concon_HCP/*.npz')[0]
# sparsify(file)
# Parallel(n_jobs=20)(delayed(sparsify)(file) for file in tqdm(glob('/data01/ayagoz/sparse_32_concon_HCP/*.npz')))
if __name__=='__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    Parallel(n_jobs=20)(delayed(sparsify)(file) for file in tqdm(glob('/data01/ayagoz/sparse_32_concon_HCP/*.npz')[start:end]))