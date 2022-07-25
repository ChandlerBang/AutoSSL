from deeprobust.graph.data import Pyg2Dpr
import argparse
import numpy as np
from meta_auto_ssl_homo import MetaSSL
from selfsl.gnn_encoder import GCN
from utils import *
import torch
import random
import time
st = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_lambda', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=0.0000)
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--fix_weight', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--ssl', type=str)
parser.add_argument('--all_clusters', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
print('===========')
print(args)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

data = get_dataset(args.dataset, args.normalize_features)
nfeat = data.features.shape[1]
args.nclass = max(data.labels) + 1

set_of_ssl = ['Par', 'Clu', 'DGI', 'PairwiseDistance', 'PairwiseAttrSim']

if data.adj.shape[0] > 5000:
    set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
else:
    set_of_ssl = set_of_ssl

encoder = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout, nlayers=args.num_layers, activation='prelu')
if args.dataset == 'arxiv':
    encoder = GCN(nfeat=nfeat, nhid=args.hidden, dropout=args.dropout, nlayers=2, activation='prelu', with_bn=True, with_res=True)


auto_agent = MetaSSL(data, encoder, set_of_ssl, args, fix_weight=args.fix_weight)
args.epochs = 3000 if args.dataset in ['computers'] else 1000
x = auto_agent.pretrain(patience=1000)

print('Time: ', time.time()-st)

auto_agent.evaluate_pretrained(x)
# if args.dataset == 'arxiv':
#     auto_agent.evaluate_pretrained_arxiv(x)
# else:
#     auto_agent.evaluate_pretrained(x)

