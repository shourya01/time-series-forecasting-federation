import os, sys
import torch
import torch.nn as nn
from typing import Union, List, Tuple

# Autoregressive LSTM

class LSTMAR(nn.Module):
    
    def __init__(
        self,
        x_size: int,
        y_size: int,
        u_size: int,
        s_size: int,
        hidden_size: int = 20,
        num_layers: int = 2,
        fcnn_sizes: Union[List,Tuple] = (20,10,10,1),
        activation: nn.Module = nn.ReLU,
        lookback: int = 8, # this will not be used, but keeping it here for consistency
        lookahead: int = 4,
        dtype: torch.dtype = torch.float32
    ):
        
        super(LSTMAR,self).__init__()
        
        # sanity checks
        assert fcnn_sizes[0] == hidden_size, "Size mismatch in first layer of FCNN."
        assert fcnn_sizes[-1] == y_size, "Size mismatch in last layer of FCNN."
        assert lookahead > 0, "Cannot have non-positive lookahead."
        
        # save values for use outside init
        self.input_size = x_size + y_size + u_size + s_size
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.dtype = dtype
        
        # lstm
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = True,
            batch_first = True,
            dropout = 0.0,
            bidirectional = False,
            proj_size = 0,
            device = None,
            dtype = dtype
        )
        
        # lstm_2
        self.lstm_2 = nn.LSTM(
            input_size = u_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = True,
            batch_first = True,
            dropout = 0.0,
            bidirectional = False,
            proj_size = 0,
            device = None,
            dtype = dtype
        )
        
        # fcnn
        self.fcnn = nn.Sequential(
            *([elem for sublist in [[nn.Linear(fcnn_sizes[i],fcnn_sizes[i+1],bias=True,dtype=dtype),activation()] for i in range(len(fcnn_sizes)-2)] for elem in sublist]+
               [nn.Linear(fcnn_sizes[-2],fcnn_sizes[-1],bias=True,dtype=dtype)])
            
        )
        
        # lookahead
        self.lookahead = lookahead
        
    def init_h_c_(self, B, device):
        
        shape = (self.num_layers,B,self.hidden_size)
        h = torch.zeros(shape,dtype=self.dtype,device=device)
        c = torch.zeros(shape,dtype=self.dtype,device=device)
        
        return h,c
    
    def forward(self,x):
        
        # extract components
        y_past, x_past, u_past, s_past, u_future, _ = x
        inp = torch.cat([y_past,x_past,u_past,s_past],dim=2)
        B, dev = inp.shape[0], inp.device
        
        # sanity check
        assert inp.shape[2] == self.input_size, "Feature dimension mismatch!"
        
        # generate states
        h,c = self.init_h_c_(B, dev)
        
        # iterate 
        for tidx in range(inp.shape[1]):
            _, (h,c) = self.lstm(inp[:,[tidx],:],(h,c))
        h_collector = []
        for tidx in range(self.lookahead):
            _, (h,c) = self.lstm_2(u_future[:,[tidx],:],(h,c))
            h_collector.append(h[-1,:,:])
        
        # pass through fcnn and return
        return torch.cat([self.fcnn(h)[:,None,:] for h in h_collector],dim=1)