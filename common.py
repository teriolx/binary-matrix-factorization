import time
import torch
import matplotlib.pyplot as plt


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