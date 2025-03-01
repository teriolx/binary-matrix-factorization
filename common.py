import time
import torch


def construct_adjacency_matrix(data):
    n_nodes = data.x.shape[0]
    n_edges = data.edge_index.shape[1]
    s = torch.sparse_coo_tensor(data.edge_index, 
                                [1 for _ in range(n_edges)], 
                                (n_nodes, n_nodes))
    return s.to_dense().numpy()


def time_wrapper(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        duration = time.time() - start
        return duration, *result
    return wrapper
