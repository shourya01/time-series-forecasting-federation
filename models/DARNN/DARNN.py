import os, sys
import torch
import torch.nn as nn
from typing import Union, List, Tuple

class DARNN(nn.Module):
    
    def __init__(
        self,
        x_size: int,
        y_size: int,
        u_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        encoder_num_layers: int,
        decoder_num_layers: int,
        lookback: int,
        lookahead: int,
        dtype: torch.dtype
    ):
        
        super(DARNN,self).__init__()
        
        # sanity checks
        assert lookback > 0, "Cannot have non positive lookback."
        assert lookahead > 0, "Cannot have non-positive lookahead."
        
        # save values for use outside init
        self.x_size, self.y_size = x_size, y_size
        self.encoder_hidden_size, self.decoder_hidden_size = encoder_hidden_size, decoder_hidden_size
        self.lookahead, self.lookback = lookahead, lookback
        self.dtype = dtype
        
        # encoder lstm for past features
        self.e_lstm_1 = nn.LSTM(
            input_size = x_size + u_size,
            hidden_size = encoder_hidden_size,
            num_layers = encoder_num_layers,
            bias = True,
            batch_first = True,
            dropout = 0.0,
            bidirectional = False,
            proj_size = 0,
            device = None,
            dtype = dtype
        )
        
        # encoder lstm for past features
        self.e_lstm_2 = nn.LSTM(
            input_size = u_size,
            hidden_size = encoder_hidden_size,
            num_layers = encoder_num_layers,
            bias = True,
            batch_first = True,
            dropout = 0.0,
            bidirectional = False,
            proj_size = 0,
            device = None,
            dtype = dtype
        )
        
        # decoder lstm
        self.lstm = nn.LSTM(
            input_size = y_size,
            hidden_size = decoder_hidden_size,
            num_layers = decoder_num_layers,
            bias = True,
            batch_first = True,
            dropout = 0.0,
            bidirectional = False,
            proj_size = 0,
            device = None,
            dtype = dtype
        )
        
        # input attention
        self.attn_inp = nn.Sequential(
            nn.Linear(in_features = 2*encoder_hidden_size + lookback, out_features = lookback, bias = False, dtype = dtype),
            nn.Tanh(),
            nn.Linear(in_features = lookback, out_features = 1, bias = False, dtype = dtype)
        )
        
        # temporal attention
        self.attn_tmp = nn.Sequential(
            nn.Linear(in_features = 2*decoder_hidden_size + encoder_hidden_size, out_features = encoder_hidden_size, bias = False, dtype = dtype),
            nn.Tanh(),
            nn.Linear(in_features = encoder_hidden_size, out_features = 1, bias = False, dtype = dtype)
        )
        
    def init_h_c_(self, B, device):
        
        shape = (self.num_layers,B,self.hidden_size)
        h = torch.zeros(shape,dtype=self.dtype,device=device)
        c = torch.zeros(shape,dtype=self.dtype,device=device)
        
        return h,c
    
    def forward(self,x):
        
        # extract components
        y_past, x_past, u_past, s_past, _ = x
        inp = torch.cat([y_past,x_past,u_past,s_past],dim=2)
        B, dev = inp.shape[0], inp.device
        
        # sanity check
        assert inp.shape[2] == self.input_size, "Feature dimension mismatch!"
        
        # generate states
        h,c = self.init_h_c_(B, dev)
        
        # iterate 
        h_collector = []
        for tidx in range(inp.shape[1]):
            _, (h,c) = self.lstm(inp[:,[tidx],:],(h,c))
            h_collector.append(h[-1,:,:])
        h_appended = torch.cat(h_collector,dim=1)
        
        # pass through fcnn and return
        return self.fcnn(h_appended)