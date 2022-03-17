import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
from numba import njit
import networkx as nx
from sklearn.cluster import KMeans
import metis
import os
from deeprobust.graph import utils

class Par(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid, device, **kwargs):
        super(Par, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.device = device
        if kwargs['args'].dataset in ['citeseer']:
            self.nparts = 1000
        elif kwargs['args'].dataset in ['photo', 'computers']:
            self.nparts = 100
        elif kwargs['args'].dataset in ['wiki']:
            self.nparts = 20
        else:
            self.nparts = 400
        pseudo_labels = self.get_label(self.nparts)
        self.pseudo_labels = pseudo_labels.to(device)
        self.disc = nn.Linear(nhid, self.nparts)
        self.sampled_indices = (self.pseudo_labels >= 0) # dummy sampling

    def make_loss(self, embeddings):
        embeddings = self.disc(embeddings)
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output[self.sampled_indices], self.pseudo_labels[self.sampled_indices])
        return loss

    def get_label(self, nparts):
        partition_file = './saved/' + self.args.dataset + '_partition_%s.npy' % nparts
        if not os.path.exists(partition_file):
            print('Perform graph partitioning with Metis...')

            adj_coo = self.data.adj.tocoo()
            node_num = adj_coo.shape[0]
            adj_list = [[] for _ in range(node_num)]
            for i, j in zip(adj_coo.row, adj_coo.col):
                if i == j:
                    continue
                adj_list[i].append(j)

            _, partition_labels =  metis.part_graph(adj_list, nparts=nparts, seed=0)
            np.save(partition_file, partition_labels)
            return torch.LongTensor(partition_labels)
        else:
            partition_labels = np.load(partition_file)
            return torch.LongTensor(partition_labels)

