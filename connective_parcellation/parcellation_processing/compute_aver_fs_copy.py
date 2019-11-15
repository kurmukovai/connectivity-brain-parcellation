import numpy as np
from glob import glob
from tqdm import tqdm

def create_average_parcellation(labels):
    '''
    Creates average parcellation from set of labels,
    for every vertex chooses most frequent label.
    
    Parameters
    -----
    
    labels - ndarray,
     N x M array, N subject, M number of mesh vertices to label
     
    Returns
    -----
    average_label - ndarray,
     array of labels of length M
    '''
    n, m = labels.shape
    average_labels = np.zeros(m)
    for i in tqdm(range(m)):
        vals, freq = np.unique(labels[:, i], return_counts=True)
        ind = np.argmax(freq)
        average_labels[i] = vals[ind]
    return average_labels

if __name__=="__main__":
    # Average Desikan parcellation
    labels = []
    files = glob('/data01/ayagoz/sparse_32_concon_HCP/parcellations/desikan_aparc/*.npy')
    for file in tqdm(files[:213]):
        label = np.load(file, allow_pickle=True)
        labels.append(label)
    labels= np.array(labels)

    aver_labels = create_average_parcellation(labels)
    np.save('/home/kurmukov/desik_200.npy', aver_labels)

