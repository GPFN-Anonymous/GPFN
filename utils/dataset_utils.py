#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os.path as osp
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon


def DataLoader(name):
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset, dataset[0]
