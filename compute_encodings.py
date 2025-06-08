import numpy as np
import scipy as sp
import pandas as pd
from common import construct_adjacency_matrix
from lpca import decomposition_at_k
from tqdm import tqdm
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import GNNBenchmarkDataset
import sys


def format_LPCA_encoding(data):
    return np.hstack((data['U'], np.transpose(data['V'])))

def format_LPCA_encoding_fixed(data):
    return data['U']


def save_encodings(matrices, out_path):
    names = [f"idx_{i}" for i in range(len(matrices))]
    np.savez_compressed(out_path, **dict(zip(names, matrices)))


def compute_encodings(data, k, out_path, format_fun, bounds=None, V=None, n_hops=1, is_single=False, fixed_k=None):
    matrices = {}
    results = []

    for i in tqdm(range(len(data))):
        A = sp.sparse.csr_matrix(construct_adjacency_matrix(data[i]))

        while n_hops > 1:
            A = A @ A
            n_hops -= 1

        t, error, nit, res = decomposition_at_k(A, k, None, V, 5000, bounds, is_single, fixed_k)
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


def load_dataset(name):
    train, val, test = None, None, None
    if name == "ZINC":
        train = ZINC(subset=True, root='data', split='train')
        val   = ZINC(subset=True, root='data', split='val')
        test  = ZINC(subset=True, root='data', split='test')
    elif name == "CIFAR":
        train = GNNBenchmarkDataset(name='CIFAR10', root='data', split='train')
        val   = GNNBenchmarkDataset(name='CIFAR10', root='data', split='val')
        test  = GNNBenchmarkDataset(name='CIFAR10', root='data', split='test')    
    
    if train is not None and val is not None and test is not None:
        return train + val + test 
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the dataset name.")
        exit()
    
    if len(sys.argv) < 3:
        print("Please provide the bounds range.")
        exit()

    if len(sys.argv) < 4:
        print("Please provide the dimension k.")
        exit()

    name = sys.argv[1]
    data = load_dataset(name)
    if sys.argv[2] == "None":
        bounds = None
    else:
        bounds = (-int(sys.argv[2]), sys.argv[2])
    k = int(sys.argv[3])
    n_hops = 1
    is_single=False
    fixed_k = None
    
    if len(sys.argv) > 4:
        n_hops = int(sys.argv[4])

    if len(sys.argv) > 5:
        is_single = sys.argv[5]

    if len(sys.argv) > 6 and sys.argv[6]:
        fixed_k = -1+2*np.random.random(size=k*k).reshape((k, k))

    # lb, ub = bounds
    # N = 40
    # k = 16
    # V = lb+ub+1*np.random.random(size=N*k).reshape((k, N))
    #compute_encodings(data, 2, 'lpca_out/lpca2b_4_4', format_LPCA_encoding, bounds)

    # python compute_encodings.py ZINC 4 16 1 False True

    print(fixed_k.shape)

    out_path = f'lpca_out/lpca_{name}_{k}_b{bounds[1] if bounds is not None else "N"}_{n_hops}hop_single{is_single}_fixedK{fixed_k is not None}'
    print("Computing LPCA:", out_path)
    
    compute_encodings(data, k, out_path, 
                      format_LPCA_encoding if fixed_k is None else format_LPCA_encoding_fixed, 
                      bounds, None, n_hops, is_single, fixed_k)
    