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
from tqdm import tqdm
import torch.nn.functional as F
import utils
from auto_ssl import AutoNodeSSL, EmptyData
import math
from torch import Tensor
from typing import List


class MetaSSL(AutoNodeSSL):

    def __init__(self, data, encoder, set_of_ssl, args, device='cuda', **kwargs):
        kwargs['fix_weight'] = 1
        super(MetaSSL, self).__init__(data, encoder, set_of_ssl, args, device='cuda', **kwargs)
        self.ssl_agent = []
        self.processed_data = EmptyData()

        self.linear = nn.Linear(1, len(set_of_ssl), bias=False).to(device)
        self.linear.weight.data.fill_(1/self.n_tasks)
        self.weight = self.linear.weight
        self.weight_optimizer = optim.Adam(list(self.linear.parameters()),
                                lr=args.lr_lambda,
                                weight_decay=0)

        self.setup_ssl(set_of_ssl)

    def setup_ssl(self, set_of_ssl):
        # initialize them
        args = self.args
        self.process_data()

        for ix, ssl in enumerate(set_of_ssl):
            agent = eval(ssl)(data=self.data,
                    processed_data=self.processed_data,
                    encoder=self.encoder,
                    nhid=self.args.hidden,
                    device=self.device,
                    args=args).to(self.device)
            self.ssl_agent.append(agent)

    def pretrain(self, patience=1e5, verbose=True):
        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm

        init_weights = deepcopy(self.encoder.state_dict())
        disc_weights = []
        for ix, ssl in enumerate(self.ssl_agent):
            if ssl.disc is not None:
                disc_weights.append(deepcopy(ssl.disc.state_dict()))

        best_homo = 0
        best_loss = 1e5
        pat = 0
        for out_i in range(self.args.epochs):
            meta_grad = 0
            self.weight_optimizer.zero_grad()

            if out_i == 0:
                self.encoder.load_state_dict(init_weights)
                params = list(self.encoder.parameters())
                self.encoder_optimizer = optim.Adam(params,
                                            lr=self.args.lr,
                                            weight_decay=self.args.weight_decay)

                params = []
                for ix, ssl in enumerate(self.ssl_agent):
                    if ssl.disc is not None:
                        ssl.disc.load_state_dict(disc_weights[ix])
                        params = params + list(ssl.disc.parameters())

                self.ssl_optimizer = optim.Adam(params,
                                            lr=self.args.lr,
                                            weight_decay=self.args.weight_decay)

            self.encoder_optimizer.zero_grad()
            self.ssl_optimizer.zero_grad()
            self.encoder.train()
            x = self.encoder(features, adj_norm)

            loss = 0
            for ix, ssl in enumerate(self.ssl_agent):
                loss_ssl = ssl.make_loss(x)
                loss = loss + loss_ssl * self.weight[ix]

            loss.backward(retain_graph=False, create_graph=True)
            param_updates = mystep(self.encoder_optimizer, self.weight)

            self.encoder.eval()
            x = self.encoder(features, adj_norm)
            loss_outer, homo = get_loss_homo(x,
                    self.data.adj.nonzero(), self.args)
            l_w_grads = torch.autograd.grad(loss_outer,
                    list(self.encoder.parameters()), retain_graph=False)
            l_w_grads = torch.cat([g.view(-1) for g in l_w_grads])
            intermidate_grad = torch.autograd.grad(param_updates, self.weight, grad_outputs=l_w_grads)[0]
            meta_grad = meta_grad + intermidate_grad
            self.ssl_optimizer.step()

            print_freq = 100
            if homo > best_homo:
                best_homo = homo
                best_weights = deepcopy(self.encoder.state_dict())
                pat = 0
            else:
                pat += 1

            if pat == patience:
                print('Early Stopped at Epoch %s' % out_i)
                break

            if out_i % print_freq == 0:
                print("Epoch: {0}, Outer loss: {1:.4f}, homo: {2:.5f}".format(out_i, loss_outer.item(), homo))
                if out_i % print_freq == 0:
                    print('Current weight:', [w.item() for w in self.weight])

            self.linear.weight.grad.copy_(meta_grad)
            self.weight_optimizer.step()
            self.linear.weight.data.clamp_(min=0, max=1)
            if out_i % print_freq == 0:
                print('Updated weight:', [w.item() for w in self.weight])

            if out_i == 1000:
                print('==== 1000 epoch...')
            if out_i == 2000:
                print('==== 2000 epoch...')

        self.encoder.eval()
        self.encoder.load_state_dict(best_weights)
        print('==== final epoch...')
        with torch.no_grad():
            x = self.encoder(features, adj_norm)
        return x.detach()



def mystep(optimizer, loss_weight):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the updates.
    """

    for group in optimizer.param_groups:
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_sums = []
        max_exp_avg_sqs = []
        state_steps = []

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = optimizer.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

        beta1, beta2 = group['betas']
        g_lambda = adam(params_with_grad,
               grads,
               exp_avgs,
               exp_avg_sqs,
               max_exp_avg_sqs,
               state_steps,
               group['amsgrad'],
               beta1,
               beta2,
               group['lr'],
               group['weight_decay'],
               group['eps'],
               loss_weight)
    return g_lambda

def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         loss_weight: List[Tensor]):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    updates = []
    updated_exp_avg = []
    updated_exp_avg_sq = []
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
        exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
        else:

            denom = ((exp_avg_sq+1e-32).sqrt() / math.sqrt(bias_correction2)).add(eps)

        step_size = lr / bias_correction1
        update = -step_size * exp_avg/denom
        updates.append(update)

        updated_exp_avg.append(exp_avg)
        updated_exp_avg_sq.append(exp_avg_sq)


    all_updates = torch.cat([u.view(-1) for u in updates])
    for i, param in enumerate(params):
        with torch.no_grad():
            exp_avgs[i].copy_(updated_exp_avg[i])
            exp_avg_sqs[i].copy_(updated_exp_avg_sq[i])
            param.add_(updates[i])
    return all_updates


from sklearn.cluster import KMeans
def get_loss_homo(x, edge_index, args):
    if args.dataset in ['arxiv', 'computers']:
        ncluster = 10
    else:
        ncluster = 5
    x_numpy = x.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=ncluster, random_state=0, n_jobs=8).fit(x_numpy)
    cluster_labels = kmeans.labels_
    homo = (cluster_labels[edge_index[0]] == cluster_labels[edge_index[1]])
    centroids = torch.FloatTensor(kmeans.cluster_centers_).to('cuda')
    logits = []
    for c in centroids:
        logits.append((-torch.square(x - c).sum(1)/1e-3).view(-1, 1))
    logits = torch.cat(logits, axis=1)
    probs = F.softmax(logits, dim=1)
    loss = F.l1_loss(probs[edge_index[0]], probs[edge_index[1]])
    return loss, np.mean(homo)
