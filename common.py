import time
import torch
import matplotlib.pyplot as plt
import numpy as np


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


def plot_nodes_error_k(data, k_list):
    plt.figure(figsize=(18, 9))
    for k in k_list:
        col_name = "k_" + str(k) 
        plt.errorbar(data.index, data[col_name]["mean"], yerr=data[col_name]["std"], fmt='-o', capsize=0.2, capthick=1, label=col_name)
        if len(data.index) > 200:
            plt.xticks(range(data.index.min(), data.index.max(), len(data.index) // 50), rotation=60)
        else:
            plt.xticks(data.index, rotation=60)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Relative Reconstruction Error")
    plt.legend()
    plt.show()


def neighbourhood_symmetric_difference(u_neigh, v_neigh):
    return np.sum(np.bitwise_xor(u_neigh, v_neigh))


def measure_encoding_similarity(A, encodings):
    similarity = {}
    n_nodes = A.shape[0]

    for v in range(n_nodes):
        for w in range(n_nodes):
            if v == w:
                continue
            d = neighbourhood_symmetric_difference(A[v], A[w])
            
            if not d in similarity:
                similarity[d] = []
            similarity[d].append(np.linalg.norm(encodings[v] - encodings[w]))
    
    return similarity



if __name__ == "__main__":
    u = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    v = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    w = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    assert neighbourhood_symmetric_difference(u, v) == 3
    assert neighbourhood_symmetric_difference(u, w) == 3
    assert neighbourhood_symmetric_difference(w, v) == 2
    assert neighbourhood_symmetric_difference(v, v) == 0