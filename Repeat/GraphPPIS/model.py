######################################## import area ########################################

# common library
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.nn.parameter import Parameter

######################################## function area ########################################

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, variant=True, residual=True):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        self.residual = residual
        
        self.in_features = 2 * in_features if variant else in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, graphs, lamda, alpha, layer_idx):
        theta = min(1, math.log(lamda/layer_idx + 1))
        graphs.update_all(fn.u_mul_e('h', 'x', 'm'), fn.sum('m', 'h'))
        if self.variant:
            support = torch.cat([graphs.ndata['h'], graphs.ndata['i']], dim=1)
            r = (1 - alpha) * graphs.ndata['h'] + alpha * graphs.ndata['i']
        else:
            support = (1 - alpha) * graphs.ndata['h'] + alpha * graphs.ndata['i']
            r = support
        graphs.ndata['h'] = theta * (support @ self.weight) + (1 - theta) * r
        # speed up convergence of the training process
        if self.residual:
            graphs.ndata['h'] = graphs.ndata['h'] + graphs.ndata['i']
        return graphs.ndata['h']
        

class GraphPPIS(nn.Module):
    
    def __init__(self, node_features, hidden_features, output_features, 
                 num_layers, dropout, lamda, alpha, variant=True, residual=True):
        super(GraphPPIS, self).__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphConvolution(hidden_features, hidden_features, variant, residual))
        
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(node_features, hidden_features))
        self.fcs.append(nn.Linear(hidden_features, output_features))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        
    def forward(self, graphs, device):
        
        graphs.ndata['i'] = F.dropout(graphs.ndata['x'], self.dropout, training=self.training)
        graphs.ndata['i'] = self.act_fn(self.fcs[0](graphs.ndata['i']))
        graphs.ndata['h'] = graphs.ndata['i'].clone()
        
        for idx, conv in enumerate(self.convs):
            graphs.ndata['h'] = F.dropout(graphs.ndata['h'], self.dropout, training=self.training)
            graphs.ndata['h'] = self.act_fn(conv(graphs, self.lamda, self.alpha, idx + 1))
        
        graphs.ndata['h'] = F.dropout(graphs.ndata['h'], self.dropout, training=self.training)
        output = self.fcs[-1](graphs.ndata['h'])
        
        graphs.ndata.pop('h')
        graphs.ndata.pop('i')
        
        return output
    




