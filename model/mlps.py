import torch
import torch.nn as nn
import torch.nn.functional as NF
import math

class PositionalEncoding(nn.Module):
    def __init__(self, L):
        """ L: number of frequency bands """
        super(PositionalEncoding, self).__init__()
        self.L= L
        
    def forward(self, inputs):
        L = self.L
        encoded = [inputs]
        for l in range(L):
            encoded.append(torch.sin((2 ** l * math.pi) * inputs))
            encoded.append(torch.cos((2 ** l * math.pi) * inputs))
        return torch.cat(encoded, -1)


def mlp(dim_in, dims, dim_out):
    """ create an MLP in format: dim_in->dims[0]->...->dims[-1]->dim_out"""
    lists = []
    dims = [dim_in] + dims
    
    for i in range(len(dims)-1):
        lists.append(nn.Linear(dims[i],dims[i+1]))
        lists.append(nn.ReLU(inplace=True))
    lists.append(nn.Linear(dims[-1], dim_out))
    
    return nn.Sequential(*lists)


class ImplicitMLP(nn.Module):
    """ implicit voxel feature MLP"""
    def __init__(self, D, C, S, dim_out, dim_enc):
        """
        Args:
            D: depth of the MLP
            C: intermediate feature dim
            S: skip layers
            dim_out: out feature dimension
            dim_enc: frequency band of positional encdoing for the vertex location
        """
        super(ImplicitMLP, self).__init__()
        self.input_ch = dim_enc * 2 * 3 + 3
       
        self.D = D
        self.C = C
        self.skips = S
        self.dim_out = dim_out
        
        self.point_encode = PositionalEncoding(dim_enc)
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.C)] + 
            [
                nn.Linear(self.C, self.C) 
                if i not in self.skips 
                else nn.Linear(self.C + self.input_ch, self.C) 
                for i in range(1, self.D) 
            ]
        )
        self.feature_linear = nn.Linear(self.C, self.dim_out)
        
    def forward(self, x):
        """ 
        Args:
            x: Bx3 3D vertex locations
        Return:
            Bx3 corresponding feature vectors
        """
        points = self.point_encode(x)
        p = points
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                p = torch.cat([points, p], 1)
            p = l(p)
            p = NF.relu(p)
        feature = self.feature_linear(p)
        return feature
