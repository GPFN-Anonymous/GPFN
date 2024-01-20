#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

"""

import argparse
from utils.dataset_utils import DataLoader
from utils.matrix_utils import *
from utils.utils import random_planetoid_splits
from models.GNN_models import *
from models.filters import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np



def RunExp(args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    Gamma_0 = args.alpha
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of trainable parameters: {total_params}")
    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()[0]
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet','GPRGNN','GIN','MLP'],
                        default='GCN')
    parser.add_argument('--missing-rate', type=float,default=0.3)
    parser.add_argument('--filter', type=str,default='empty')
    args = parser.parse_args()
    print(args)
    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name =="GIN":
        Net = GIN_Net
    elif gnn_name == "MLP":
        Net = GCN_Net

    filter_name = args.filter
    if filter_name == 'Katz':
        Filter = Katz_Filter()
    elif filter_name == 'RES':
        Filter = RES_Filter()
    elif filter_name  == 'AGE':
        Filter = AGE_Filter()
    elif filter_name == 'log':
        Filter = Log_Filter()
    elif filter_name == 'SGC':
         Filter = SGC_Filter()
    elif filter_name == 'empty':
        Filter = No_Filter()
    elif filter_name == 'scale-1':
        Filter = Scale_Filter(beta=0.8,eps=1e-3,power=1)
    elif filter_name == 'scale-2':
        Filter = Scale_Filter(beta=0.6,eps=1e-3,power=2)
    elif filter_name == 'scale-3':
        Filter = Scale_Filter(beta=0.4,eps=1e-3,power=3)
    elif filter_name=='Bernet':
        Filter = Bern_Filter()

    dname = args.dataset
    dataset, data = DataLoader(dname)
    #随机mask边
    if args.missing_rate>0:
        data.edge_index = remove_edges_symmetrically(data.edge_index,args.missing_rate)

    data.adj,data.edge_list= normalize_adj(data.edge_index,Filter,data.y.shape[0])     
    if gnn_name=="MLP":
        I = torch.eye(data.adj.shape[0])
        rows, cols = I.nonzero(as_tuple=True)
        indices = torch.stack([rows, cols], dim=0)
        values = I[rows, cols]
        data.adj = torch.sparse_coo_tensor(indices, values, I.size()).coalesce()

    if gnn_name != 'GPRGNN' and gnn_name != 'ChebNet':
        data.edge_index = data.adj                                                    
    print(data)
    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    Results0 = []

    for RP in tqdm(range(RPMAX)):

        test_acc, best_val_acc, Gamma_0 = RunExp(
            args, dataset, data, Net, percls_trn, val_lb)
        Results0.append([test_acc, best_val_acc, Gamma_0])
    print(Results0)
    test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f'{gnn_name} on dataset {args.dataset} with filter {args.filter} at missing_rate {args.missing_rate}, in {RPMAX} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
