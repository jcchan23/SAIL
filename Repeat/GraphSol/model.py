######################################## import area ########################################

# common library
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch.nn.parameter import Parameter
from dgl.nn.pytorch.glob import AvgPooling

######################################## function area ########################################

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, graphs):
        graphs.update_all(fn.u_mul_e('h', 'x', 'm'), fn.sum('m', 'h'))
        graphs.ndata['h'] = graphs.ndata['h'] @ self.weight
        if self.bias is not None:
            return graphs.ndata['h'] + self.bias
        else:
            return graphs.ndata['h']


class AttnPooling(nn.Module):
    def __init__(self, in_features, dense_features, n_heads):
        super(AttnPooling, self).__init__()
        self.in_features = in_features
        self.dense_features = dense_features
        self.n_heads = n_heads
        self.fc1 = nn.Linear(in_features, dense_features)
        self.fc2 = nn.Linear(dense_features, n_heads)
    
    def forward(self, graphs):
        with graphs.local_scope():
            graphs.ndata['heads'] = torch.tanh(self.fc1(graphs.ndata['h']))
            # (num_nodes, n_heads)
            graphs.ndata['heads'] = self.fc2(graphs.ndata['heads'])
            attns = dgl.softmax_nodes(graphs, 'heads')
            for i in range(self.n_heads):
                graphs.ndata[f'head_{i}'] = attns[:,i].reshape(-1, 1)
            result = []
            for i in range(self.n_heads):
                result.append(dgl.sum_nodes(graphs, 'h', f'head_{i}').unsqueeze(0))
            output = torch.mean(torch.cat(result, dim=0), dim=0)
            return output
        
class GraphSol(nn.Module):
    
    def __init__(self, in_features, hidden_features, output_features, attention_features, attention_heads):
        super(GraphSol, self).__init__()
        
        self.conv1 = GraphConvolution(in_features=in_features, out_features=hidden_features * 4)
        self.fc1 = nn.LayerNorm(hidden_features * 4)
        self.conv2 = GraphConvolution(in_features=hidden_features * 4, out_features=hidden_features)
        self.fc2 = nn.LayerNorm(hidden_features)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.pooling = AttnPooling(hidden_features, dense_features=attention_features, n_heads=attention_heads)
        self.fc_final = nn.Linear(hidden_features, output_features)
        
    def forward(self, graphs, device):
        
        graphs.ndata['h'] = graphs.ndata['x'].clone()
        graphs.ndata['h'] = self.conv1(graphs)
        graphs.ndata['h'] = self.relu(self.fc1(graphs.ndata['h']))
        graphs.ndata['h'] = self.conv2(graphs)
        graphs.ndata['h'] = self.relu(self.fc2(graphs.ndata['h']))
        
        output = self.pooling(graphs) 
        output = F.sigmoid(self.fc_final(output))
        graphs.ndata.pop('h')
        
        return output
    




