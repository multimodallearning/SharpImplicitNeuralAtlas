import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m, scale=30):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            hidden_ch = max(m.weight.shape[0],m.weight.shape[1])
            val = np.sqrt(6/hidden_ch)/scale if m.weight.shape[1]>4 else 1 / m.weight.shape[1]
            torch.nn.init.uniform_(m.weight,-val,val)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
                
#convlinear is work-around to have different initialisation modes for sine and relu layers
class ConvLinear(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features,out_features,1)
    def forward(self,x):
        return self.conv(x.permute(0,3,1,2)).permute(0,2,3,1)
class ReLUSiren(nn.Module):
    def __init__(self,in_features, out_features, hidden_ch=256,scale=30,num_layers=3):
        super().__init__()
        self.scale = scale
        listnet = [nn.Linear(in_features,hidden_ch)]
        for i in range(num_layers):
            listnet.append(ConvLinear(hidden_ch,hidden_ch))
        listnet.append(nn.Linear(hidden_ch,out_features))
        self.net = nn.Sequential(*listnet)
        
    def forward(self,x):
        scale = self.scale
        x = torch.sin(scale*self.net[0](x))
        x = torch.sin(scale*self.net[1](x))
        for i in range(2,len(self.net)-2):
            x = F.relu(self.net[i](x))#
        x = torch.sin(scale*self.net[-2](x))
        return self.net[-1](x)
    
class Siren(nn.Module):
    def __init__(self,in_features, out_features, hidden_ch=256,scale=30,num_layers=3):
        super().__init__()
        self.scale = scale
        listnet = [nn.Linear(in_features,hidden_ch)]
        for i in range(num_layers):
            listnet.append(nn.Linear(hidden_ch,hidden_ch))
        listnet.append(nn.Linear(hidden_ch,out_features))
        self.net = nn.Sequential(*listnet)
        
    def forward(self,x):
        scale = self.scale
        x = torch.sin(scale*self.net[0](x))
        for i in range(1,len(self.net)-1):
            x = torch.sin(scale*self.net[i](x))
        return self.net[-1](x)
    