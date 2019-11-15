import os
import numpy as np
from tqdm import tqdm
from nilearn.surface import load_surf_data, load_surf_mesh
from nibabel.freesurfer.io import read_geometry
from sklearn.neighbors import KNeighborsClassifier


def load_mesh_boris(path='/home/bgutman/datasets/HCP/Dan_iso5.m'):
    
    '''
    load boris mesh (.m) file
    faces enumerated from 1, but after loading from 0
   
    
    usage:
    vertices, faces = load_mesh_boris('/home/bgutman/datasets/HCP/Dan_iso5.m')
    '''
    with open(path, 'r') as f:
        iso5 = f.read()
    iso5 = iso5.split('\n')
    vertices = []
    faces = []
    for line in iso5:
        a = line.split(' ')
        if a[0] == 'Vertex':
            vertices.append([float(sym) for sym in a[2:5]])
        elif a[0] == 'Face':
            faces.append([int(sym) for sym in a[2:]])
    vertices = np.array(vertices)
    faces = np.array(faces) - 1
    return vertices, faces


def transfer_mesh_color(subject_id,
                        atlas='aparc',
                        reconall_folder='/data01/ayagoz/HCP_1200/FS_reconall/',
                        concon_mesh='/home/kurmukov/HCP/Dan_iso5.m'):
    '''
    Transfer mesh labels from subject sphere mesh, to concon sphere mesh
    
    Parameters:
    
    subject_id - int,
     subject id
     
    atlas - str,
     atlas to transfer, possible values: aparc (Desikan-Killiany), aparc.a2009s (Destrieux Atlas).
     Defined by free surfer https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
     
    reconall_folder - str,
     path to recon-all FS output
     
    concon_mesh - str,
     path to ConCon sphere mesh
    '''

    lh_vertices, _ = read_geometry(f'{reconall_folder}{subject_id}/surf/lh.sphere.reg')
    rh_vertices, _ = read_geometry(f'{reconall_folder}{subject_id}/surf/rh.sphere.reg')

    lh_labels = load_surf_data(f'{reconall_folder}{subject_id}/label/lh.{atlas}.annot')
    rh_labels = load_surf_data(f'{reconall_folder}{subject_id}/label/rh.{atlas}.annot')
    
    lh_vertices /= 100
    rh_vertices /= 100
    
    lh_vertices_CC, _ = load_mesh_boris(concon_mesh)
    rh_vertices_CC, _ = load_mesh_boris(concon_mesh)
    
    knn = KNeighborsClassifier(n_neighbors=5,
                           weights='uniform',
                           metric='minkowski')
    
    knn.fit(lh_vertices, lh_labels)
    lh_labels_CC = knn.predict(lh_vertices_CC)
    
    knn.fit(rh_vertices, rh_labels)
    rh_labels_CC = knn.predict(rh_vertices_CC)
    
    rh_labels_CC[rh_labels_CC != -1] += np.max(lh_labels_CC)
    
    labels_CC = np.concatenate([lh_labels_CC, rh_labels_CC])

    return labels_CC


if __name__ == '__main__':
    save_folder_desikan = '/data01/ayagoz/sparse_32_concon_HCP/parcellations/desikan_aparc'
    save_folder_destrieux = '/data01/ayagoz/sparse_32_concon_HCP/parcellations/destrieux_aparc2009'
    reconall_folder = '/data01/ayagoz/HCP_1200/FS_reconall/'
    
    for sid in tqdm(os.listdir(reconall_folder)):
        labels_desikan = transfer_mesh_color(sid, atlas='aparc')
        labels_destrieux = transfer_mesh_color(sid, atlas='aparc.a2009s')
        np.save(f'{save_folder_desikan}/{sid}.npy', labels_desikan)
        np.save(f'{save_folder_destrieux}/{sid}.npy', labels_destrieux)
    
    
    