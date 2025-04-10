import numpy as np
import scipy as sp
from common import construct_adjacency_matrix
from lpca import decomposition_at_k
from tqdm import tqdm
from torch_geometric.datasets import ZINC
from scipy.optimize import Bounds


def format_LPCA_encoding(data):
    return np.hstack((data['U'], np.transpose(data['V'])))


def compute_encoding(graph, k, format_fun):
    A = sp.sparse.csr_matrix(construct_adjacency_matrix(graph))

    _, _, _, res = decomposition_at_k(A, k)
    return format_fun(res)


def save_encodings(matrices, out_path):
    names = [f"idx_{i}" for i in range(len(matrices))]
    np.savez_compressed(out_path, **dict(zip(names, matrices)))


def compute_encodings(data, k, out_path, format_fun, bounds=None):
    matrices = {}

    for i in tqdm(range(len(data))):
        A = sp.sparse.csr_matrix(construct_adjacency_matrix(data[i]))

        _, _, _, res = decomposition_at_k(A, k)
        matrices[f"idx_{i}"] = format_fun(res)
    
    np.savez_compressed(out_path, **matrices)


if __name__ == "__main__":
    zinc_train = ZINC(subset=True, root='data', split='train')
    zinc_val   = ZINC(subset=True, root='data', split='val')
    zinc_test  = ZINC(subset=True, root='data', split='test')

    data = list(zinc_train) + list(zinc_val) + list(zinc_test)    
    compute_encodings(data, 8, 'lpca_out/lpca8b.npz', format_LPCA_encoding, bounds=Bounds(-4, 3))
