# Architecture from https://github.com/czq142857/BAE-NET

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.02)
        x = F.leaky_relu(self.l2(x), 0.02)
        return self.l3(x)
    
class conv3DNet(nn.Module):
    def __init__(self, lcode_size):
        super(conv3DNet, self).__init__()
        self.lcode_size = lcode_size
        self.conv1 = nn.Conv3d(1, 32, (4, 4, 4), stride=(2,2,2), padding = 1, bias = True)
        self.conv2 = nn.Conv3d(32, 64, (4, 4, 4), stride=(2,2,2), padding = 1, bias = True)
        self.conv3 = nn.Conv3d(64, 128, (4, 4, 4), stride=(2,2,2), padding = 1, bias = True)
        self.conv4 = nn.Conv3d(128, 256, (4, 4, 4), stride=(2,2,2), padding = 1, bias = True)
        self.conv5 = nn.Conv3d(256, 128, (4, 4, 4), stride=(1,1,1), padding = 0, bias = True)

        self.in1 = nn.InstanceNorm3d(32, affine=True)
        self.in2 = nn.InstanceNorm3d(64, affine=True)
        self.in3 = nn.InstanceNorm3d(128, affine=True)
        self.in4 = nn.InstanceNorm3d(256, affine=True)
        
    def forward(self, x):
        x = x.view(1, 1, 64, 64, 64)
        
        x = self.conv1(x)
        x = F.leaky_relu(self.in1(x), 0.02)
        
        x = self.conv2(x)
        x = F.leaky_relu(self.in2(x), 0.02)
        
        x = self.conv3(x)
        x = F.leaky_relu(self.in3(x), 0.02)
        
        x = self.conv4(x)
        x = F.leaky_relu(self.in4(x), 0.02)
        
        x = self.conv5(x)
        x = torch.sigmoid(x)

        return x.reshape(self.lcode_size)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.zeros_(m.bias)
        
class BaeNet(nn.Module):
    def __init__(
        self, num_shapes, lcode_size = 128
    ):
        self.num_parts = num_shapes
        super(BaeNet, self).__init__()
        self.apply(weights_init)
        self.net = MLP(128 + 3, 1024, 256, num_shapes)        
        self.voxel_net = conv3DNet(lcode_size)
        self.lcode_size = lcode_size
        
    def encode(self, voxels):        
        return self.voxel_net(voxels)
        
    def forward(self, input):

        l3 = torch.sigmoid(self.net(input))
        sdf = l3.max(dim=1).values.view(-1, 1)
                                            
        return sdf, l3
