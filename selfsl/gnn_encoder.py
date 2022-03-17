import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def batch_forward(self, seq, adj, sparse=False):
        weight = self.weight.unsqueeze(0).expand(seq.size(0),
                self.weight.size(0), self.weight.size(1))
        seq_fts = torch.bmm(seq, weight)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, activation='prelu', dropout=0, nlayers=1, with_bn=False, with_res=False):
        #TODO number of layers
		# PReLU
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.act = nn.ReLU() if activation.lower() == 'relu' else nn.PReLU()
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.layers.append(GraphConvolution(nfeat, nhid))
        if with_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(nhid))
        for ix in range(nlayers-1):
            self.layers.append(GraphConvolution(nhid, nhid))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
        self.with_bn = with_bn
        self.with_res = with_res
        self.reset_parameters()

    def reset_parameters(self):
        def weight_reset(m):
            # if isinstance(m, GraphConvolution):
            #     m.reset_parameters()
            # if isinstance(m, GraphConvolution):
            #     glorot(m.weight)
            #     if m.bias is not None:
            #         zeros(m.bias)
            if isinstance(m, GraphConvolution):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

        self.apply(weight_reset)

    def forward(self, x, adj):
        # TODO: modify dropout
        if self.with_res:
            prev_x = 0
            for ix, layer in enumerate(self.layers):
                x = layer(x + prev_x, adj)
                x = self.bns[ix](x) if self.with_bn else x
                x = self.act(x)
                if ix != len(self.layers) - 1:
                    x = F.dropout(x, self.dropout, training=self.training)
                prev_x = x
        else:
            for ix, layer in enumerate(self.layers):
                x = layer(x, adj)
                x = self.bns[ix](x) if self.with_bn else x
                x = self.act(x)
                if ix != len(self.layers) - 1:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x

    # Shape of seq: (batch, nodes, features)
    def batch_forward(self, seq, adj, sparse=False):
        x = seq
        for ix, layer in enumerate(self.layers):
            x = layer.batch_forward(x, adj, sparse)
            x = self.bns[ix](x) if self.with_bn else x
            x = self.act(x)
            if ix != len(self.layers) - 1:
                x = F.dropout(x, self.dropout, training=self.training)
        return x

    def get_prob(self, x, adj):
        # TODO: modify
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                # x = F.relu(x)
                x = self.act(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


if __name__ == "__main__":
    encoder = GCN(10, nhid=10)
