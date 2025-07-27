# Binary Matrix Vectorization

This repository contains the implementation and experiments with two mathods for vectorization of binary matrices LPCA and tSVD. The aim is to compute these deompositions for molecular datasets ZINC, Peptides and the image recongnition dataset CIFAR to retrieve graph encodings. These can then be integrated into the Graph Transfomer for graph learning.

# Structure of the Repository

- output/ - some of the outputs computed and used in the analyses
- scripts/ - various bash scripts for computing encodings locally as well as on VSC
- src/
    - LPCA/ - contains the implementation `lpca.py` and various analysis in the notebooks
    - tSVD/ - contains the implementation `tSVD.py` and various analysis in the notebooks
    - `common.py` - contains common function definitions used by both methods
    - `compute_mose.py`, `compute_rwse` - code snippets from the [GraphGPS code](https://github.com/linusbao/MoSE/) for comparison with other encodings
    - `compute_encoding_similarity.py`, `encoding_similarity.ipynb` - analyzes the encodings similiary based on the neighbourhood similarity of the node pairs


# Setup

To run the encoding computation you can modify one of the example scripts from `scripts/`. The dependencies can be installed in a virtual conda environment.

```
conda create -n matrix-vectorization python=3.10
conda activate matrix-vectorization
pip install -r requirements.txt
```
