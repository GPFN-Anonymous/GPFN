import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from matplotlib.font_manager import FontProperties
import argparse
from utils.dataset_utils import DataLoader
from utils.matrix_utils import *
from utils.utils import *
from models.GNN_models import *
from models.filters import *
import torch
import numpy as np


def decompose(adj, dataset, filter,norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized


    scipy_sparse_coo = sp.coo_matrix(laplacian)

    values = scipy_sparse_coo.data
    indices = np.vstack((scipy_sparse_coo.row, scipy_sparse_coo.col))

    values = torch.from_numpy(values).float()
    indices = torch.from_numpy(indices).long()
    torch_sparse_tensor = torch.sparse_coo_tensor(indices, values, scipy_sparse_coo.shape)
    torch_sparse_tensor = Filter.filter(torch_sparse_tensor)
    values = torch_sparse_tensor.values().numpy()
    indices = torch_sparse_tensor.indices().numpy()
    norm_adj = sp.coo_matrix((values, (indices[0], indices[1])), shape=torch_sparse_tensor.size())


    evalue, evector = np.linalg.eig(norm_adj.toarray())
    np.save("./evalue/cora_raw.npy", evalue)
    print(max(evalue))
    print(min(evalue))
    evalue = evalue.real 
    counts, bin_edges = np.histogram(evalue, bins=50, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    spline = UnivariateSpline(bin_centers, counts, s=len(counts)*150)  

    # 生成平滑的曲线上的点
    smooth_bin_centers = np.linspace(bin_centers.min(), bin_centers.max(), 300)
    smooth_counts = spline(smooth_bin_centers)

    # 画直方图和频率曲线
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    # 直方图改为蓝色
    ax.hist(evalue, bins=50, facecolor='#4682B4', alpha=0.7)
    plt.xlim(left=evalue.min())
    plt.ylim(bottom=0)
    # 画出频率曲线
    ax.plot(smooth_bin_centers, smooth_counts, color='#FF7B06')

    font = {'family': 'Times New Roman',
            'color': 'black',
            #'weight': 'bold',
            'size': 28,
            }
    plt.rc('font', family='Times New Roman', size=28)
    plt.tight_layout()
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    font_prop = FontProperties(family='Times New Roman', size=25, weight='bold')
    plt.legend(['Frequency Curve', 'Frequency'], prop=font_prop)
    plt.tight_layout()
    fig.savefig("cora_katz.jpg")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--missing-rate', type=float,default=0)
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN','GIN','MLP'],
                        default='GPRGNN')
    parser.add_argument('--filter', type=str,default='empty',choices=['Katz','RES','AGE','SGC','log','scale-1','scale-2','scale-3','empty','Bernet'])
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
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
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
        Filter = Log_Filter(beta=0.65)
    elif filter_name == 'SGC':
         Filter = SGC_Filter()
    elif filter_name == 'empty':
        Filter = No_Filter()
    elif filter_name == 'scale-1':
        Filter = Scale_Filter(beta=0.5,eps=1e-3,power=1)
    elif filter_name == 'scale-2':
        Filter = Scale_Filter(beta=0.6,eps=1e-3,power=2)
    elif filter_name == 'scale-3':
        Filter = Scale_Filter(beta=0.6,eps=1e-3,power=3)
    elif filter_name=='Bernet':
        Filter = Bern_Filter()

    dname = args.dataset
    dataset, data = DataLoader(dname)

    if args.missing_rate>0:
        data.edge_index = remove_edges_symmetrically(data.edge_index,args.missing_rate)

    data.adj,data.edge_list= normalize_adj(data.edge_index,Filter,data.y.shape[0])     
    decompose(data.adj,dname,Filter)                                                 