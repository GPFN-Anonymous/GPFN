import torch
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

class No_Filter:
    def __init__(self,beta=0.05):
        self.beta=beta

    def filter(self,adj_matrix):
        return adj_matrix



class Katz_Filter:
    def __init__(self,beta=0.05,eps=1e-3):
        self.beta = beta
        self.eps = eps

    def filter(self,adj_matrix):
        adj_matrix = (adj_matrix.to_dense() @ adj_matrix.to_dense() @ adj_matrix.to_dense() @ adj_matrix.to_dense()) 
        adj_matrix = torch.inverse((torch.eye(adj_matrix.shape[0]) - self.beta * adj_matrix)) - torch.eye(
            adj_matrix.shape[0])
        adj_matrix = 1 / self.beta * (adj_matrix * (adj_matrix > self.eps)).to_sparse()
        return adj_matrix

class RES_Filter:
    def __init__(self,beta=0.7,eps=1e-3):
        self.beta = beta
        self.eps = eps

    def filter(self,adj_matrix):
        adj_matrix = torch.eye(adj_matrix.shape[0]) + adj_matrix
        for i in range(2):
            adj_matrix = self.beta * torch.mm(adj_matrix, adj_matrix)
        adj_matrix = (adj_matrix * (adj_matrix > self.eps)).to_sparse()
        return adj_matrix
            
class AGE_Filter:
    def __init__(self,dataset='cora'):
        if dataset == 'cora':
            self.lambda_dataset = 1/1.48
        elif dataset == 'citeseer':
            self.lambda_dataset = 1/1.50
        else:
            self.ambda_dataset = 1
        self.eps = 1e-3

    def filter(self,adj_matrix):
        # data.adj = torch.eye(data.adj.shape[0]) - lambda_dataset * data.adj.to_dense()      # Lsym
        adj_matrix = (1-self.lambda_dataset) * torch.eye(adj_matrix.shape[0]) + self.lambda_dataset * adj_matrix    # Asym
        adj_matrix = (adj_matrix * (adj_matrix > self.eps)).to_sparse()
        return adj_matrix

class Log_Filter:

    def __init__(self,beta=0.8,eps=1e-3):
        self.beta = beta
        self.eps = eps

    def filter(self,adj_matrix):
        adj_matrix = adj_matrix.to_dense()
        #adj_matrix = (adj_matrix.to_dense() @ adj_matrix.to_dense() @ adj_matrix.to_dense() @ adj_matrix.to_dense()) 
        # data.adj = torch.eye(data.adj.shape[0]) - lambda_dataset * data.adj.to_dense()      # Lsym
        eigenvalues, eigenvectors = scipy.linalg.eigh(self.beta*adj_matrix)
        U = torch.tensor(eigenvectors)
        A = np.diag(eigenvalues)
        A = torch.tensor(A).to_sparse()
        filter_A = torch.diag(torch.log(torch.diag(torch.inverse(torch.eye(A.size(0)) - A))))
        filter_A = filter_A *(filter_A >self.eps)
        adj_matrix = torch.sparse.mm(torch.sparse.mm(U, filter_A),U.t())
        #adj_matrix = (adj_matrix * (adj_matrix > self.eps)).to_sparse()
        adj_matrix = adj_matrix.to_sparse()
        return adj_matrix
    
class Scale_Filter:

    def __init__(self,beta=0.8,eps=1e-3,power=1):
        self.beta = beta
        self.eps = eps
        self.power = power

    def filter(self,adj_matrix):
        adj_matrix=adj_matrix.to_dense()
        #adj_matrix = (adj_matrix.to_dense() @ adj_matrix.to_dense() @ adj_matrix.to_dense() @ adj_matrix.to_dense()) 
        if self.power==1:
            adj_matrix = torch.inverse((torch.eye(adj_matrix.shape[0]) - self.beta*adj_matrix))
            #adj_matrix = (adj_matrix * (adj_matrix > self.eps)).to_sparse()
            adj_matrix = adj_matrix.to_sparse()
        elif self.power==2:
            adj_matrix = torch.inverse((torch.eye(adj_matrix.shape[0]) - self.beta*adj_matrix))
            adj_matrix = torch.sparse.mm(adj_matrix, adj_matrix)
            adj_matrix = (adj_matrix * (adj_matrix > self.eps)).to_sparse()
        elif self.power==3:
            adj_matrix = torch.inverse((torch.eye(adj_matrix.shape[0]) - self.beta*adj_matrix))
            adj_matrix_2 = torch.sparse.mm(adj_matrix, adj_matrix)
            adj_matrix = torch.sparse.mm(adj_matrix_2, 2 * adj_matrix)
            adj_matrix = (adj_matrix * (adj_matrix > self.eps)).to_sparse()
        return adj_matrix
    

class SGC_Filter:
    def __init__(self,beta=0.7,eps=1e-3):
        self.beta = beta
        self.eps = eps

    def filter(self,adj_matrix):
        adj_matrix = (adj_matrix.to_dense() @ adj_matrix.to_dense() @ adj_matrix.to_dense() @ adj_matrix.to_dense())  # larger the adjancy
        adj_matrix = (adj_matrix * (adj_matrix > self.eps)).to_sparse()
        return adj_matrix
    


class Bern_Filter:
    def __init__(self,beta=0.7,eps=1e-3):
        self.beta = beta
        self.eps = eps

    def filter(self,adj_matrix):
        adj_matrix = (torch.eye(adj_matrix.shape[0]) - 0.5*adj_matrix)
        adj_matrix = (adj_matrix @ adj_matrix @ adj_matrix @ adj_matrix @ adj_matrix)
        adj_matrix = (adj_matrix * (adj_matrix > self.eps)).to_sparse()
        return adj_matrix