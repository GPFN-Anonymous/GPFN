This is a Pytorch implementation of GPFN: Infinite-Horizon Graph Filters: Leveraging Power Series to Enhance Sparse Information Aggregation

# Infinite-Horizon Graph Filters: Leveraging Power Series to Enhance Sparse Information Aggregation

More details of the paper and dataset will be released after it is published.


# The Code

## Requirements

Following is the suggested way to install the dependencies:

    conda install --file GPFN.yml

Note that ``pytorch >=1.10``.

## Folder Structure

```tex
└── code-and-data
    ├── data                    # Including datasets-Amazon and Planetoid
    ├── models                  # The core source code of our model GPFN
    │   ├── filters.py          # Including GPFN filters and baseline filters
    │   ├── GNN_models.py       # Including baseline models of GNNs
    ├── utils                   # Defination of auxiliary functions for running
    │   ├── dataset_utils.py    # Data load and process preparation   
    │   ├── matrix_utils.py     # Matrix computation used in experiment
    │   └── utils.py            # Other anxiliary functions
    ├── draw_elegvalue.py       # Analyse and Visualise Eigenvalue
    ├── evalue                  # Including analyse result
    ├── GPFN.yml                # The python environment needed for GPFN
    ├── README.md               # This document
    ├── Reproduce.sh            # Basic reproducing script
    └── train_model.py          # Main file
```

## Datasets

Download Cora & Citeseer datasets from https://github.com/kimiyoung/planetoid; 

Download AmaComp & AmaPhoto datasets from https://github.com/shchur/gnn-benchmark.
 
 ## Main Baseline Codes
  - GPRGNN: " Adaptive Universal Generalized PageRank Graph Neural Network"  (https://github.com/jianhao2016/GPRGNN)
  - HiGCN: "Higher-order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes" (https://github.com/Yiminghh/HiGCN)
  - SGC: "Simplifying Graph Convolutional Networks" (https://github.com/Tiiiger/SGC)
  - AGE: "Adaptive Graph Encoder for Attributed Graph Embedding" (https://github.com/thunlp/AGE)
