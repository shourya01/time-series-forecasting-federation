import torch
import torch.nn as nn
import math
from math import log

# paper:
# https://proceedings.neurips.cc/paper_files/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html

class CausalConv1D(nn.Module):
    def __init__(
        self, 
        input_feature_len, 
        output_feature_len, 
        kernel_size, # free parameter
        dtype
        ):
        
        super(CausalConv1D, self).__init__()
        
        self.padding = (kernel_size - 1) # Calculate padding to preserve sequence length
        self.conv1 = nn.Conv1d(in_channels=input_feature_len, 
                               out_channels=output_feature_len, 
                               kernel_size=kernel_size, 
                               stride=1, 
                               padding=self.padding, 
                               dilation=1,
                               dtype = dtype)

    def forward(self, x):
        # expected shape of x: (batch_size,seq_len,input_feature_len)
        x = x.transpose(1, 2)  # Swap `seq_len` and `input_feature_len`
        x = self.conv1(x)
        x = x[:, :, :-self.padding]  # Remove padding from the end
        x = x.transpose(1, 2)  # Swap back to original dimensions
        return x
    
class LogSparseAttentionTransformer(nn.Module):
    def __init__(
        self, 
        num_layers, # free parameter
        d_model, # free parameter
        nhead, # free parameter
        seq_len, 
        window_size, # free parameter: around 5 or 10ish
        dtype
        ):
        
        super(LogSparseAttentionTransformer, self).__init__()
        
        self.seq_len = seq_len
        self.dtype = dtype
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True, dtype=dtype)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.register_buffer('log_sparse_mask', self.generate_causal_log_sparse_mask(seq_len, window_size))

    def generate_causal_log_sparse_mask(self, seq_len, window_size):
        mask = torch.full((seq_len, seq_len), float('inf'), dtype=self.dtype)
        for i in range(seq_len):
            start = max(0, i - window_size)
            for j in range(start, i + 1):
                mask[i, j] = 0
            for j in range(0, int(math.log2(i + 1)) + 1):
                step = 2**j
                if i - step >= 0:
                    mask[i, i - step] = 0
        return mask

    def forward(self, src):
        output = self.transformer_encoder(src, mask=self.log_sparse_mask)
        return output
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, dtype=torch.float32):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=dtype) * -(log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class LogTransAR(nn.Module):
    
    def __init__(
        self,
        x_size: int,
        y_size: int,
        u_size: int,
        s_size: int,
        lookback: int = 8,
        lookahead: int = 4,
        model_dim: int = 64,
        num_heads: int = 4,
        kernel_size: int = 6,
        window_size: int = 5,
        dtype: torch.dtype = torch.float32
    ):
        
        super(LogTransAR, self).__init__()
        
        self.inp_size = x_size + y_size + u_size + s_size
        self.out_size = y_size
        self.seq_len_input = lookback
        self.seq_len_forecast = lookahead
        self.dtype = dtype
        
        # number of transformer layers using the rule from the paper
        num_encoder_layers = int(math.floor(math.log2(self.seq_len_input))) + 1
        
        # create the modules
        self.input_encoder = CausalConv1D(
            input_feature_len = self.inp_size,
            output_feature_len = model_dim,
            kernel_size = kernel_size,
            dtype = self.dtype
        )
        self.transformer = LogSparseAttentionTransformer(
            num_layers = num_encoder_layers,
            d_model = model_dim,
            nhead = num_heads,
            seq_len = self.seq_len_input,
            window_size = window_size,
            dtype = self.dtype
        )
        self.output_decoder = nn.Linear(
            in_features = model_dim,
            out_features = self.out_size,
            dtype = self.dtype
        )
        
        self.pos_embed = PositionalEmbedding(
            d_model = model_dim,
            max_len = self.seq_len_input,
            dtype = self.dtype
        )
        
    def forward(self, x):
        
        # extract data
        y_past, x_past, u_past, s_past, _ , _ = x
        
        # create relevant tensors
        src = torch.cat([y_past,x_past,u_past,s_past], dim=-1)
        tokens = self.input_encoder(src) # initally the tokens are those derived from the input
        tokens = tokens + self.pos_embed(tokens)
        
        # asserts
        assert src.shape[1] == self.seq_len_input
        
        outputs = []
        for _ in range(self.seq_len_forecast):
            out_tokens = self.transformer(tokens)
            outputs.append(out_tokens[:,[-1],:])
            tokens = torch.cat([tokens[:,1:,:],outputs[-1]], dim=1)
            tokens = tokens + self.pos_embed(tokens)
            
        return self.output_decoder(torch.cat(outputs,dim=1))