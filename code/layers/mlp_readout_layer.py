import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, m):
        return self.l2(F.relu(self.l1(m)))
    
class EGCNApply(nn.Module):
    def __init__(self, in_feats):
        super(EGCNApply, self).__init__()
        self.mlp = MLP(in_feats, in_feats //2, in_feats // 4)
        self.linear = nn.Linear(in_feats //4, 1)
        self.max_subds = 2
        
    def forward(self, edge):
        return {
            "m": #torch.sigmoid(
                self.linear(
                    F.relu(self.mlp(edge.data["m"]))
                )
            #) * self.max_subds
        }

class EdgeMLPReadout(nn.Module):
    def __init__(self, in_dim):
        super(EdgeMLPReadout, self).__init__()
        self.apply_mod = EGCNApply(in_dim)

    def forward(self, g, h):
        g.ndata['h'] = h
        g.apply_edges(fn.u_add_v('h','h','m'))
        g.apply_edges(func=self.apply_mod)
        return g.edata["m"].squeeze()
