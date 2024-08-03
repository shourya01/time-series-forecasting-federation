import torch
import torch.nn as nn

# paper:
# https://arxiv.org/abs/1704.02971

# Function to parse inputs

def parse_inputs(inp, lookback, lookahead, y_size, x_size, u_size, s_size):
    
    split_sizes = [y_size,x_size,u_size,s_size,y_size,u_size]
    y_past, x_past, u_past, s_past, y_future, u_future = torch.split(inp,split_sizes,dim=-1)
    
    if lookback>lookahead:
        y_future, u_future = y_future[:,:lookahead,:], u_future[:,:lookahead,:]
    if lookahead>lookback:
        y_past, x_past, u_past, s_past = y_past[:,:lookback,:], x_past[:,:lookback,:], u_past[:,:lookback,:], s_past[:,:lookback,:]
        
    # return in the format
    return y_past, x_past, u_past, s_past, u_future, y_future

class DARNN(nn.Module):
    
    def __init__(
        self,
        x_size: int,
        y_size: int,
        u_size: int,
        s_size: int,
        encoder_hidden_size: int = 20,
        decoder_hidden_size: int = 20,
        encoder_num_layers: int = 2,
        decoder_num_layers: int = 2,
        lookback: int = 8,
        lookahead: int = 4,
        dtype: torch.dtype = torch.float32
    ):
        
        super(DARNN,self).__init__()
        
        # sanity checks
        assert lookback > 0, "Cannot have non positive lookback."
        assert lookahead > 0, "Cannot have non-positive lookahead."
        
        # save values for use outside init
        self.x_size, self.y_size, self.u_size, self.s_size = x_size, y_size, u_size, s_size
        self.encoder_hidden_size, self.decoder_hidden_size = encoder_hidden_size, decoder_hidden_size
        self.encoder_num_layers, self.decoder_num_layers = encoder_num_layers, decoder_num_layers
        self.input_size = x_size + s_size
        self.lookahead, self.lookback = lookahead, lookback
        self.dtype = dtype
        
        # encoder lstm for past features
        self.e_lstm = nn.LSTM(
            input_size = x_size + s_size,
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
        self.d_lstm = nn.LSTM(
            input_size = y_size + u_size + encoder_hidden_size,
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
        
        # projection
        self.proj = nn.Linear(in_features = decoder_hidden_size, out_features = y_size, bias = False, dtype = dtype)
        
    def init_h_c_enc_(self, B, device):
        
        shape = (self.encoder_num_layers,B,self.encoder_hidden_size)
        h = torch.zeros(shape,dtype=self.dtype,device=device)
        c = torch.zeros(shape,dtype=self.dtype,device=device)
        
        return h,c
    
    def init_h_c_dec_(self, B, device):
        
        shape = (self.decoder_num_layers,B,self.decoder_hidden_size)
        h = torch.zeros(shape,dtype=self.dtype,device=device)
        c = torch.zeros(shape,dtype=self.dtype,device=device)
        
        return h,c
    
    def forward(self,x):
        
        # extract components
        y_past, x_past, u_past, s_past, u_future, _ = parse_inputs(x,self.lookback,self.lookahead,self.y_size,self.x_size,self.u_size,self.s_size)
        inp = torch.cat([x_past,s_past],dim=2)
        inpT = inp.permute(0,2,1)
        B, dev = inp.shape[0], inp.device
        
        # sanity check
        assert inp.shape[1] == self.lookback, "Time dimension mismatch in input!"
        assert inp.shape[2] == self.input_size, "Feature dimension mismatch in input!"
        
        # generate states
        h_e,c_e = self.init_h_c_enc_(B, dev)
        
        # iterate 
        h_e_collector = []
        for tidx in range(self.lookback):
            # generate attention input
            att_enc_in = torch.cat([inpT,h_e[-1,:,:][:,None,:].repeat(1,self.input_size,1),
                        c_e[-1,:,:][:,None,:].repeat(1,self.input_size,1)],dim=-1)
            att_enc = self.attn_inp(att_enc_in)[:,:,0][:,None,:]
            _, (h_e,c_e) = self.e_lstm(inp[:,[tidx],:]*att_enc,(h_e,c_e))
            h_e_collector.append(h_e[-1,:,:][:,None,:])
        h_e_appended = torch.cat(h_e_collector,dim=1)
        
        # decoder input
        inp_d_0 = torch.cat([y_past,u_past],dim=2)
        
        # generate states
        h_d,c_d = self.init_h_c_dec_(B, dev)
        
        # generate states 
        for tidx in range(self.lookback):
            # generate attention input
            att_dec_in = torch.cat([h_e_appended,h_d[-1,:,:][:,None,:].repeat(1,self.lookback,1),
                        c_d[-1,:,:][:,None,:].repeat(1,self.lookback,1)],dim=-1)
            att_dec = self.attn_tmp(att_dec_in)[:,:,0][:,:,None].repeat(1,1,self.encoder_hidden_size)
            dec_in = torch.cat([inp_d_0[:,[tidx],:],(att_dec*h_e_appended).sum(dim=1)[:,None,:]],dim=-1)
            _, (h_d,c_d) = self.d_lstm(dec_in,(h_d,c_d))
            if tidx == self.lookback - 1:
                y_last = self.proj(h_d[-1,:,:])
                
        # future states
        y_pred = []
        for tidx in range(self.lookahead):
            att_dec_in = torch.cat([h_e_appended,h_d[-1,:,:][:,None,:].repeat(1,self.lookback,1),
                        c_d[-1,:,:][:,None,:].repeat(1,self.lookback,1)],dim=-1)
            att_dec = self.attn_tmp(att_dec_in)[:,:,0][:,:,None].repeat(1,1,self.encoder_hidden_size)
            dec_in = torch.cat([y_last[:,None,:],u_future[:,[tidx],:],(att_dec*h_e_appended).sum(dim=1)[:,None,:]],dim=-1)
            _, (h_d,c_d) = self.d_lstm(dec_in,(h_d,c_d))
            y_last = self.proj(h_d[-1,:,:])
            y_pred.append(y_last[:,None,:])
            
        # concatenate and return
        return torch.cat(y_pred,dim=1)