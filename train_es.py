import cma
from deeprobust.graph.data import Pyg2Dpr
import argparse
import numpy as np
from auto_ssl import AutoNodeSSL
from selfsl.gnn_encoder import GCN
from utils import *
from sklearn.metrics import roc_auc_score
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0000)
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--fix_weight', type=int, default=1)
parser.add_argument('--debug', type=int, default=0)
args = parser.parse_args()

import logging
LOG_FILENAME = f'logs_es/{args.dataset}_{args.seed}.log'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.warn(args)

torch.cuda.set_device(args.gpu_id)
print('===========')
print(args)

# random seed setting
np.random.seed(args.seed)
data = get_dataset(args.dataset, args.normalize_features)
nfeat = data.features.shape[1]

set_of_ssl = ['Par', 'Clu', 'DGI', 'PairwiseDistance', 'PairwiseAttrSim']
opts = cma.CMAOptions()
opts.set('bounds', [(0,) * len(set_of_ssl), (1,) * len(set_of_ssl) ])
opts.set('popsize', 8)
opts.set('seed', args.seed)
opts.set('maxiter', 100)

es = cma.CMAEvolutionStrategy([0.5] * len(set_of_ssl), 0.5, opts)
logging.debug(set_of_ssl)

def get_homo(x, data):
    ncluster = 10 if args.dataset in ['arxiv', 'computers'] else 5
    print('perform clustering with KMeans...')
    x = x.cpu().numpy()
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=ncluster, random_state=0, n_jobs=8).fit(x)
    cluster_labels = kmeans.labels_
    edge_index = data.adj.nonzero()
    homo = (cluster_labels[edge_index[0]] == cluster_labels[edge_index[1]])
    print('embedding homo:', np.mean(homo))
    return 1-np.mean(homo)

def fitness(values):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.dataset == 'arxiv':
        local_set_of_ssl = ['Clu', 'DGISample', 'PairwiseDistance', 'PairwiseAttrSim']
        encoder = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout, nlayers=args.num_layers, activation='relu', with_bn=True)
    else:
        encoder = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout)
    if data.adj.shape[0] > 5000:
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl
    auto_agent = AutoNodeSSL(data, encoder, local_set_of_ssl, args, fix_weight=args.fix_weight)
    auto_agent.set_weight(values)
    x = auto_agent.pretrain(patience=20, verbose=False)
    acc, std, nmi = auto_agent.evaluate_pretrained(x)
    loss = get_homo(x, data)
    logging.debug(f'current weights {values}' + \
            f'\nhomo loss: {loss}' + \
            f'\nAcc and std: {acc} {std}' + \
            f'\nNMI: {nmi}')
    print('homo loss:', loss)
    return loss

if __name__ == "__main__":
    if args.debug:
        es.optimize(fitness, n_jobs=0, min_iterations=100)
    else:
        torch.multiprocessing.set_start_method('spawn')# good solution !!!!
        es.optimize(fitness, n_jobs=1, min_iterations=100)
        es.result_pretty()


