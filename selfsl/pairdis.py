import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
from numba import njit
import networkx as nx
import os
from deeprobust.graph import utils
from numba.typed import List
from torch_geometric.utils import negative_sampling
from torch_geometric.data import NeighborSampler
from multiprocessing import Pool


class PairwiseDistance(nn.Module):
    """
    Faster sampling
    """
    def __init__(self, data, processed_data, encoder, nhid, device, **kwargs):
        super(PairwiseDistance, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        self.device = device
        self.nclass = 4
        self.disc = nn.Linear(nhid, self.nclass)
        self.build_distance()
        self.cnt = 0

    def build_distance(self):
        if self.args.dataset == 'arxiv':
            A = self.data.adj
            self.num_nodes = A.shape[0]
            A_2 = A @ A
            A_2_aug = A + A_2 + sp.eye(self.num_nodes)
            self.A_3_aug = A_2_aug
            self.A_list = [A, A_2]
            self.pos_edge_index = torch.LongTensor(np.array(A_2_aug.nonzero()))
        else:
            A = self.data.adj
            self.num_nodes = A.shape[0]
            A_2 = A @ A
            A_3 = A_2 @ A
            A_3_aug = A + A_2 + A_3 + sp.eye(self.num_nodes)
            self.A_3_aug = A_3_aug
            self.A_list = [A, A_2, A_3]
            self.pos_edge_index = torch.LongTensor(np.array(A_3_aug.nonzero()))

    def multi_sample(self, n, kn, A_list, all):
        params = [(n, kn, A_list, all), (n, kn, A_list, all),
                (n, kn, A_list, all), (n, kn, A_list, all)]

        pool = Pool(processes=4)
        data = pool.map(work_wrapper, params)
        pool.close()
        pool.join()
        return data

    def sample(self, n=256):
        labels = []
        sampled_edges = []

        runs = 4
        kn = 1000
        all_target_nodes = np.random.default_rng().choice(self.num_nodes,
                        runs*n, replace=False).astype(np.int32)
        multi = False
        if multi:
            data = self.multi_sample(n, kn, self.A_list, all_target_nodes)
            sampled_edges = [torch.cat([d[1] for d in data], axis=1)]
            labels = [torch.cat([d[0] for d in data])]
            ii = 2
        else:
            for _ in range(runs):
                for ii, A in enumerate(self.A_list):
                    target_nodes = all_target_nodes[ii*n: ii*n+n]
                    A = A[target_nodes]
                    if A.nnz > 1e5:
                        A = A[:, all_target_nodes]
                    edges = torch.LongTensor(A.nonzero())
                    num_edges = edges.shape[1]

                    if num_edges <= kn:
                        idx_selected = np.arange(num_edges)
                    else:
                        idx_selected = np.random.default_rng().choice(num_edges,
                                        kn, replace=False).astype(np.int32)
                    labels.append(torch.ones(len(idx_selected), dtype=torch.long) * ii)
                    sampled_edges.append(edges[:, idx_selected])
                    kn = len(idx_selected)

        if self.num_nodes > 5000: # if the graph is large, we use more runs
            runs = 10
        if self.cnt % runs == 0:
            self.cnt += 1
            pos_edge_index = torch.LongTensor(self.A_3_aug[all_target_nodes].nonzero())
            neg_edges = negative_sampling(
                        edge_index=pos_edge_index, num_nodes=self.num_nodes,
                        num_neg_samples=kn*n)

            self.neg_edges = neg_edges
            neg_edges = self.neg_edges
        neg_edges = self.neg_edges
        sampled_edges.append(neg_edges)
        labels.append(torch.ones(neg_edges.shape[1], dtype=torch.long) * (ii+1))

        labels = torch.cat(labels).to(self.device)
        sampled_edges = torch.cat(sampled_edges, axis=1)
        return sampled_edges, labels

    def make_loss(self, embeddings):
        node_pairs, labels = self.sample()

        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = self.disc(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, labels)
        return loss

    def get_label(self):
        graph = nx.from_scipy_sparse_matrix(self.data.adj)

        if not os.path.exists(f'saved/node_distance_{self.args.dataset}.npy'):
            path_length = dict(nx.all_pairs_shortest_path_length(graph, cutoff=self.nclass-1))
            distance = - np.ones((len(graph), len(graph))).astype(int)

            for u, p in path_length.items():
                for v, d in p.items():
                    distance[u][v] = d

            distance[distance==-1] = distance.max() + 1
            distance = np.triu(distance)
            np.save(f'saved/node_distance_{self.args.dataset}.npy', distance)
        else:
            print('loading distance matrix...')
            distance = np.load(f'saved/node_distance_{self.args.dataset}.npy')
        self.distance = distance
        return torch.LongTensor(distance) - 1


def work_wrapper(args):
    return worker(*args)

def worker(n, kn, A_list, all_target_nodes):
    labels = []
    sampled_edges = []
    for ii, A in enumerate(A_list):
        target_nodes = all_target_nodes[ii*n: ii*n+n]
        A = A[target_nodes]
        edges = torch.LongTensor(A.nonzero())
        num_edges = edges.shape[1]

        if num_edges <= kn:
            idx_selected = np.arange(num_edges)
        else:
            idx_selected = np.random.default_rng().choice(num_edges,
                            kn, replace=False).astype(np.int32)
        labels.append(torch.ones(len(idx_selected), dtype=torch.long) * ii)
        sampled_edges.append(edges[:, idx_selected])
        kn = len(idx_selected)
    labels = torch.cat(labels)
    sampled_edges = torch.cat(sampled_edges, axis=1)
    return labels, sampled_edges
