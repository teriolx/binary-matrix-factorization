from common import measure_encoding_similarity, construct_adjacency_matrix
from torch_geometric.datasets import ZINC
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    train = ZINC(subset=True, root='data', split='train')
    val = ZINC(subset=True, root='data', split='val')
    test = ZINC(subset=True, root='data', split='test')

    in_name = 'lpca_ZINC_32_b4_1hop_singleFalse_fixedKTrue'

    encoding = np.load("lpca_out/" + in_name + ".npz")

    results = []

    for i in tqdm(range(len(encoding))):
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
    
    pd.DataFrame(results).to_parquet("similarity_test_" + in_name + '.parquet')