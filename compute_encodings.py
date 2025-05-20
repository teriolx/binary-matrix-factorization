import numpy as np
import scipy as sp
import pandas as pd
from common import construct_adjacency_matrix
from lpca import decomposition_at_k
from tqdm import tqdm
from torch_geometric.datasets import ZINC


def format_LPCA_encoding(data):
    return np.hstack((data['U'], np.transpose(data['V'])))

def format_LPCA_encoding_fixed(data):
    return data['U']

def compute_encoding(graph, k, format_fun):
    A = sp.sparse.csr_matrix(construct_adjacency_matrix(graph))

    _, _, _, res = decomposition_at_k(A, k)
    return format_fun(res)


def save_encodings(matrices, out_path):
    names = [f"idx_{i}" for i in range(len(matrices))]
    np.savez_compressed(out_path, **dict(zip(names, matrices)))


def compute_encodings(data, k, out_path, format_fun, bounds=None, V=None):
    matrices = {}
    results = []

    for i in tqdm(range(len(data))):
        A = sp.sparse.csr_matrix(construct_adjacency_matrix(data[i]))

        t, error, nit, res = decomposition_at_k(A, k, None, V, 5000, bounds)
        matrices[f"idx_{i}"] = format_fun(res)
        results.append(
            {
                "graph_id": i,
                "n_nodes": data[i].x.shape[0],
                "nit": nit,
                "error": error,
                "time": t
            }
        )
    
    np.savez_compressed(out_path + '.npz', **matrices)
    pd.DataFrame(results).to_parquet(out_path + '.parquet')


if __name__ == "__main__":
    train = ZINC(subset=True, root='data', split='train')
    val   = ZINC(subset=True, root='data', split='val')
    test  = ZINC(subset=True, root='data', split='test')

    data = train + val + test 
    
    bounds = (-4, 4)
    # lb, ub = bounds
    # N = 40
    # k = 16
    # V = lb+ub+1*np.random.random(size=N*k).reshape((k, N))
    compute_encodings(data, 2, 'lpca_out/lpca2b_4_4', format_LPCA_encoding, bounds)
    