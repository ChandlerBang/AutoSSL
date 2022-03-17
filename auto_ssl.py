import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from selfsl import *
from deeprobust.graph.utils import to_tensor, normalize_adj_tensor, accuracy
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (f1_score, roc_auc_score, adjusted_rand_score,
        accuracy_score, average_precision_score, v_measure_score)
from tqdm import tqdm
import torch.nn.functional as F
import utils

class EmptyData:

    def __init__(self):
        self.adj_norm = None
        self.features = None
        self.labels = None

class AutoNodeSSL(nn.Module):

    def __init__(self, data, encoder, set_of_ssl, args, device='cuda', **kwargs):
        super(AutoNodeSSL, self).__init__()
        self.args = args
        self.encoder = encoder.to(device)
        self.data = data
        self.adj = data.adj
        self.features = data.features
        self.device = device
        self.n_tasks = len(set_of_ssl)
        self.fix_weight = kwargs['fix_weight']
        if kwargs['fix_weight']:
            self.weight = torch.ones(len(set_of_ssl)).to(device)
        else:
            self.linear = nn.Linear(1, len(set_of_ssl)-1, bias=False).to(device)
            self.weight = self.linear.weight
            self.reset_parameters()

        self.ssl_agent = []
        self.optimizer = None
        self.processed_data = EmptyData()
        self.setup_ssl(set_of_ssl)

    def reset_parameters(self):
        # self.weight.data.fill_(1/(self.weight.size(1)+1))
        self.linear.weight.data.fill_(1/self.n_tasks)

    def set_weight(self, values):
        self.weight = torch.FloatTensor(values).to(self.device)

    def setup_ssl(self, set_of_ssl):
        # initialize them
        args = self.args
        self.process_data()

        params = list(self.encoder.parameters())
        for ix, ssl in enumerate(set_of_ssl):
            agent = eval(ssl)(data=self.data,
                    processed_data=self.processed_data,
                    encoder=self.encoder,
                    nhid=self.args.hidden,
                    device=self.device,
                    args=args).to(self.device)
            self.ssl_agent.append(agent)
            if agent.disc is not None:
                params = params + list(agent.disc.parameters())

            if hasattr(agent, 'gcn2'):
                params = params + list(agent.gcn2.parameters())

        self.optimizer = optim.Adam(params,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

    def process_data(self):
        # still needed
        if self.processed_data.adj_norm is None:
            features, adj, labels = to_tensor(self.data.features,
                    self.data.adj, self.data.labels, device=self.device)
            adj_norm = normalize_adj_tensor(adj, sparse=True)
            self.processed_data.adj_norm = adj_norm
            self.processed_data.features = features
            self.processed_data.labels = labels

    def pretrain(self, patience=1e5, verbose=True):
        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm
        if self.args.epochs == 0:
            with torch.no_grad():
                x = self.encoder(features, adj_norm)
            return x.detach()

        best_loss = 1e5
        pat = 0
        for i in range(self.args.epochs):
            self.optimizer.zero_grad()
            x = self.encoder(features, adj_norm)

            loss = self.get_combined_loss(x)
            if i % 50 == 0 and verbose:
                print(f'Epoch {i}: {loss.item()}')
            if loss < best_loss:
                best_loss = loss
                best_weights = deepcopy(self.encoder.state_dict())
                pat = 0
            else:
                pat += 1
            if pat == patience:
                print('Early Stopped at Epoch %s' % i)
                break
            loss.backward()
            self.optimizer.step()

        self.encoder.eval()
        self.encoder.load_state_dict(best_weights)
        with torch.no_grad():
            x = self.encoder(features, adj_norm)
        return x.detach()

    def get_combined_loss(self, x):
        loss = 0
        for ix, ssl in enumerate(self.ssl_agent):
            loss = loss + self.weight[ix] * ssl.make_loss(x)
        return loss

    def evaluate_pretrained(self, x):
        args = self.args
        idx_train = self.data.idx_train
        idx_val = self.data.idx_val
        idx_test = self.data.idx_test
        labels = self.processed_data.labels
        xent = nn.CrossEntropyLoss()

        runs = 10 if args.dataset != 'wiki' else 20
        accs = []
        val_accs = []
        for _ in range(runs):

            if args.dataset == 'wiki':
                split_id = _
                idx_train = self.data.splits['train'][split_id]
                idx_val = self.data.splits['val'][split_id]
                idx_test = self.data.splits['test'][split_id]

            log = LogReg(x.shape[1], labels.max().item()+1).to(self.device)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0)
            best_acc_val = 0

            if args.dataset in ['photo', 'computers', 'corafull']:
                epochs = 3000
            else:
                epochs = 300

            if args.dataset in ['citeseer']:
                opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.01)

            for _ in range(epochs):
                log.train()
                opt.zero_grad()
                output = log(x[idx_train])
                loss = xent(output, labels[idx_train])
                loss.backward()
                opt.step()

            log.eval()
            logits = log(x[idx_test])
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == labels[idx_test]).float() / labels[idx_test].shape[0]
            accs.append(acc.item() * 100)

        print('Average accuracy:', np.mean(accs), np.std(accs))

        kmeans_input = x.cpu().numpy()
        nclass = self.data.labels.max().item()+1
        kmeans = KMeans(n_clusters=nclass, random_state=0, n_jobs=8).fit(kmeans_input)
        pred = kmeans.predict(kmeans_input)
        labels = self.data.labels
        nmi = v_measure_score(labels, pred)
        print('Node clustering:', nmi)
        edge_index = self.data.adj.nonzero()
        homo = (pred[edge_index[0]] == pred[edge_index[1]])
        print('Homo with knowledge of cluster num:', np.mean(homo))
        return np.mean(accs), np.std(accs), nmi

def reset_mlp(m):
    nn.init.xavier_uniform_(m.weight.data)
    if m.bias is not None:
        m.bias.data.fill_(0.0)

class LogReg(nn.Module):
    def __init__(self, nfeat, nclass):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(nfeat, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

