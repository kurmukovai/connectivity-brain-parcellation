import numpy as np
from scipy.sparse import coo_matrix
from igraph import Graph
from igraph import ADJ_MAX

def compute_partition(adj):
    '''
    Computes partition of the graph using Louvain algorithm
    
    Parameters
    -------
    
    graph - igraph Graph
    
    Returns
    -------
    
    labels - ndarray
    '''
    try:
        graph = Graph.Weighted_Adjacency(adj.tolist(), mode=ADJ_MAX, attr='weight')
        levels = graph.community_multilevel(weights='weight', return_levels=True)
        levels = np.array([level.membership for level in levels]).astype(int)
        labels = np.array(levels[-1])
    except BaseException as e:
        print(e)
        labels = np.array([-1]) 
    return labels

def generate_subgraphs(adjacency, partition, min_subgraph_size=2):
    '''
    Generates subgraphs based on partition
    
    Parameters
    -------
    
    adjacency - ndarray,
     adjacecy matrix of a graph
     
    partition - ndarray,
     clustering labels of graph's nodes
     
    min_graph_size - int,
     minimal size subgraph
     (used to pass isolated nodes)
     
    Returns
    -------
    
    adjacency_list - list,
     list of adjacency matrices of generated subgraphs
     
    colors - ndarray,
     array of unique cluster labels,
     excluding rare clusters (len(cluster)<min_graph_size)
     
    rare_colors - ndarray,
     array of unique cluster labels,
     of rare clusters (len(cluster)<min_graph_size)

    '''
    colors, freq = np.unique(partition, return_counts=True)
    rare_colors = colors[freq < min_subgraph_size]
    colors = colors[freq >= min_subgraph_size]
    
    adjacencies_list = [adjacency[partition==color, :][:, partition==color]
                     for color in colors]
    
    return adjacencies_list, colors, rare_colors


def cluster_subgraphs(adjacency, partition, min_subgraph_size=10, min_subgraph_to_split=30):
    '''
    Cluster subgraphs of a graph
    
    Parameters
    --------
    
    adjacency - ndarray,
     adjacecy matrix of a graph
     
    partition - ndarray,
     clustering labels of graph's nodes
     
    min_graph_size - int,
     minimal size subgraph
     (used to pass isolated nodes)
     
    min_subgraph_to_split - int,
     minimal size subgraph to split
     (used to restrict small subgraphs to be further clustered)
     
    Returns
    -------
    
    partition_at_level - ndarray,
     hierarchical graph partitioning
    '''
    adjacencies_list, colors, rare_colors = generate_subgraphs(adjacency, partition, min_subgraph_size)
    partition_at_level = np.zeros_like(partition)

    if rare_colors.size != 0:
        for c in rare_colors:
            partition_at_level[np.where(partition == c)] = -1

    max_label = 0

    for adj, c in zip(adjacencies_list, colors):
        subgraph_size = adj.shape[0]
        if subgraph_size < min_subgraph_to_split:
            labels = np.zeros(subgraph_size)
        else:
            labels = compute_partition(adj)
        # to be consistent in labeling accross different subgraphs
        partition_at_level[np.where(partition == c)[0]] = labels + max_label
        max_label += labels.max() + 1

    return partition_at_level