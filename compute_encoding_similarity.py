from common import measure_encoding_similarity, construct_adjacency_matrix
from torch_geometric.datasets import ZINC
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys


if __name__ == "__main__":
    train = ZINC(subset=True, root='data', split='train')
    val = ZINC(subset=True, root='data', split='val')
    test = ZINC(subset=True, root='data', split='test')

    dir_path = sys.argv[1]
    file_name = sys.argv[2]

    encoding = np.load(dir_path + file_name + ".npz")

    results = []

    for i in tqdm(range(len(train))):
        sim_measures = measure_encoding_similarity(construct_adjacency_matrix(train[i]), encoding[f"idx_{i}"])
        for d, similarities in sim_measures.items():
            for s in similarities:
                results.append(
                    {
                        "graph_id": i,
                        "d": d,
                        "sim": s
                    }
                )
    
    pd.DataFrame(results).to_parquet("similarity_res/similarity_" + file_name + '.parquet')