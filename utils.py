import os.path as osp
from torch_geometric.datasets import Planetoid, PPI, WikiCS, Coauthor, Amazon, CoraFull
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset, PrePtbDataset
import scipy.sparse as sp
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from deeprobust.graph import utils
from deeprobust.graph.utils import get_train_val_test_gcn, get_train_val_test
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import add_remaining_self_loops, to_undirected


def get_attacked_dataset_node_embed(name, normalize_features, transform=None):
    data = Dataset(root='attacked_data', name=name, seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    model = NodeEmbeddingAttack()
    model.attack(adj, attack_type="add", n_candidates=10000, n_perturbations=2000, dim=128)
    modified_adj = model.modified_adj
    data.adj = modified_adj
    if normalize_features:
        data.features = utils.normalize_feature(features.todense())
    return data

def get_attacked_dataset(name, normalize_features):
    name, atk_method = name.split('_')
    print(f'loading {atk_method} attck for dataset {name}...')
    if atk_method == 'embed':
        return get_attacked_dataset_node_embed(name, normalize_features)

    data = Dataset(root='attacked_data', name=name, seed=15)
    adj, features, labels = data.adj, data.features, data.labels

    if atk_method == 'gfa':
        # data.features = features.todense()
        data.features = utils.normalize_feature(features.todense())
        return data

    perturbed_data = PrePtbDataset(root='attacked_data/',
                        name=name,
                        attack_method=atk_method,
                        ptb_rate=0.15)
    data.adj = perturbed_data.adj
    data.features = features.todense()
    if normalize_features:
        data.features = utils.normalize_feature(features.todense())
    return data

def get_dataset(name, normalize_features=False, transform=None, if_dpr=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['corafull']:
        dataset = CoraFull(path)
    elif name in ['arxiv', 'proteins']:
        dataset = PygNodePropPredDataset(name='ogbn-'+name)
    elif name in ['cs', 'physics']:
        dataset = Coauthor(path, name)
    elif name in ['computers', 'photo']:
        dataset = Amazon(path, name)
    elif name in ['wiki']:
        dataset = WikiCS(root='data/')
        dataset.name = 'wiki'
    elif name in ['acm', 'blogcatalog', 'flickr', 'polblogs']:
        # dataset = Dataset(root='data/', name=name, setting='gcn', seed=15)
        dataset = Dataset(root='data/', name=name, setting='gcn')
        if normalize_features:
            dataset.features = utils.normalize_feature(dataset.features.todense())
        else:
            dataset.features = dataset.features.todense()
        return dataset
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    if not if_dpr:
        return dataset

    if name == 'wiki':
        # return Pyg2Dpr(dataset, multi_splits=True)
        data =  Pyg2Dpr(dataset, multi_splits=True)
        # data.idx_train, data.idx_val, data.idx_test = get_train_val_test_gcn(data.labels)
        return data

    else:
        return Pyg2Dpr(dataset)


class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, multi_splits=False, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name

        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if dataset_name == 'ogbn-arxiv': # symmetrization
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        # enable link prediction ....
        # self.enable_link_prediction(pyg_data)

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape
        if not multi_splits:
            if hasattr(pyg_data, 'train_mask'):
                # for fixed split
                # self.idx_train, self.idx_val, self.idx_test = get_train_val_test_gcn(self.labels)
                # We don't use fixed splits in this project...
                self.idx_train = mask_to_index(pyg_data.train_mask, n)
                self.idx_val = mask_to_index(pyg_data.val_mask, n)
                self.idx_test = mask_to_index(pyg_data.test_mask, n)
                self.name = 'Pyg2Dpr'
            else:
                try:
                    # for ogb
                    self.idx_train = splits['train']
                    self.idx_val = splits['valid']
                    self.idx_test = splits['test']
                    self.name = 'Pyg2Dpr'
                except:
                    # for other datasets
                    # self.idx_train, self.idx_val, self.idx_test = get_train_val_test_gcn(self.labels)
                    self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                            nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)
        else:
            # For wiki
            self.splits = self.load_splits(pyg_data)
            self.idx_train = self.splits['train'][0] # by default, it is from the first split
            self.idx_val = self.splits['val'][0]
            self.idx_test = self.splits['test'][0]
            self.name = 'Pyg2Dpr'

    def load_splits(self, data):
        splits = {'train': [], 'val': [], 'test': []}
        n = data.num_nodes
        for i in range(0, data.train_mask.shape[1]):
            train_mask = data.train_mask[:, i]
            val_mask = data.val_mask[:, i]
            if len(data.test_mask.shape) == 1:
                test_mask = data.test_mask
            else:
                test_mask = data.test_mask[:, i]
            idx_train = mask_to_index(train_mask, n)
            idx_val = mask_to_index(val_mask, n)
            idx_test = mask_to_index(test_mask, n)

            splits['train'].append(idx_train)
            splits['val'].append(idx_val)
            splits['test'].append(idx_test)
        return splits


    def enable_link_prediction(self, pyg_data):
        # pyg_data = train_test_split_edges(pyg_data)
        pyg_data = train_test_split_edges(pyg_data, val_ratio=0.1, test_ratio=0.4)
        self.train_pos_edge_index = pyg_data.train_pos_edge_index
        # self.train_neg_edge_index = pyg_data.train_neg_edge_index
        self.val_pos_edge_index = pyg_data.val_pos_edge_index
        self.val_neg_edge_index = pyg_data.val_neg_edge_index
        self.test_pos_edge_index = pyg_data.test_pos_edge_index
        self.test_neg_edge_index = pyg_data.test_neg_edge_index


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def get_perturbed_adjnorm(adj, perturbation):
    adj = adj.tolil()
    idx = eval(perturbation)
    idx_T = (idx[1] , idx[0])
    adj[idx] = 1 - adj[idx]
    adj[idx_T] = 1 - adj[idx_T]
    ptb_adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    # ptb_adj_norm = utils.normalize_adj_tensor(ptb_adj, sparse=True)
    ptb_adj_norm = utils.normalize_sparse_tensor(ptb_adj)
    return ptb_adj_norm

def get_tsne(x, data, args):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import pandas as pd
    labels = data.labels[data.idx_test]
    x = x.cpu().numpy()
    model = TSNE(n_components=2, random_state=0)
    res = model.fit_transform(x)
    res = res[data.idx_test]

    fig, ax = plt.subplots()
    groups = pd.DataFrame(res, columns=['x', 'y']).assign(category=labels).groupby('category')
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name, s=2)
        # ax.legend()
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    plt.savefig(f'may24logs_homo_meta/{args.dataset}_{args.ssl}.pdf', bbox_inches='tight')

    # res = model.fit_transform(x)
    # plt.scatter(res[50:,0] , res[50:,1])

