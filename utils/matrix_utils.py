#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from models.GNN_models import *
from models.filters import *
from deeprobust.graph.global_attack import Random
import random
import torch
from scipy.sparse import csr_matrix
import numpy as np



def get_degree(edge_list):
    row = edge_list[0]
    deg = torch.bincount(row)
    return deg

def normalize_adj(edge_list, Filter,num_n=1024,rate=0, seed=1024, r_type='flip'):        # adj, add, remove, flip
    if r_type != 'none':
        adj = torch.sparse.FloatTensor(edge_list, torch.ones(edge_list.shape[1])).coalesce()
        after_attack_adj = attack_adj(rate, seed, adj, r_type='flip')

        edge_list = torch.Tensor(after_attack_adj.todense()).to_sparse().indices()



    deg = get_degree(edge_list)  # N*1
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]          # D^{-1/2} A D^{-1/2}
    norm_adj = torch.sparse.FloatTensor(edge_list, v,size=(num_n, num_n)).coalesce() ##this is Asym
    print(norm_adj.shape)
    norm_adj = Filter.filter(norm_adj)

    # re:  I - D^{-1/2} A D^{-1/2}  (have added self-loop before)
    # norm_adj = torch.eye(norm_adj.shape[0]).to_sparse() - norm_adj
    return norm_adj, edge_list

def attack_adj(ptb_rate, seed, adj, r_type='flip'):
    adj = adj.to_dense()
    adj = torch.where(adj>0, torch.ones_like(adj), torch.zeros_like(adj))
    adj = csr_matrix(adj)
    attacker = Random()
    random.seed(seed)
    n_perturbations = int(ptb_rate * (adj.sum() // 2))
    attacker.attack(adj, n_perturbations, type=r_type)
    perturbed_adj = attacker.modified_adj
    return perturbed_adj



def remove_edges_symmetrically(edge_list, removal_ratio):
    """
    Remove edges from an undirected graph symmetrically.

    :param edge_list: numpy array of shape (2, N) representing N undirected edges.
    :param removal_ratio: proportion of edges to remove.
    :return: numpy array of the reduced edge list.
    """
    # Ensure the smaller index is always first to represent undirected edges
    ordered_edges = np.sort(edge_list, axis=0)

    # Create a unique set of edges
    unique_edges = np.unique(ordered_edges, axis=1)

    # Calculate the number of edges to keep
    num_edges_to_keep = int((1 - removal_ratio) * unique_edges.shape[1])

    # Randomly select edges to keep
    indices_to_keep = np.random.choice(unique_edges.shape[1], num_edges_to_keep, replace=False)
    kept_edges = unique_edges[:, indices_to_keep]

    # Reconstruct the original undirected edge list format
    reduced_edge_list = np.concatenate([kept_edges, np.flip(kept_edges, axis=0)], axis=1)

    return torch.from_numpy(reduced_edge_list)
