import torch 
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# xLSTM paper: https://arxiv.org/abs/2405.04517

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

class BlockDiagonalLinear(nn.Module):
    def __init__(self, dim, block_size=4, scale=1, **kwargs):
        super(BlockDiagonalLinear, self).__init__()
        
        self.dim = dim
        self.block_size = block_size
        self.scale = scale

        self.num_full_blocks = dim // block_size
        self.last_block_size = dim % block_size

        self.blocks = nn.ModuleList([
            nn.Linear(block_size, block_size, bias=False, **kwargs) for _ in range(self.num_full_blocks)
        ])

        if self.last_block_size > 0:
            self.blocks.append(nn.Linear(self.last_block_size, self.last_block_size, bias=False, **kwargs))

    def forward(self, x):
        x_split = torch.split(x, self.block_size, dim=-1)
        x_transformed = [block(x_block) for x_block, block in zip(x_split, self.blocks)]
        return self.scale*torch.cat(x_transformed, dim=-1)

class Swish(nn.Module):
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class CustomConv1D(nn.Module):
    def __init__(self, **kwargs):
        super(CustomConv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, **kwargs)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.pad(x, (2, 1))
        x = self.conv1d(x)
        x = x.squeeze(1)
        
        return x

class mLSTMCell(nn.Module):
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dtype: torch.dtype = torch.float32
    ):
        
        super(mLSTMCell,self).__init__()
        
        upscaled_dim = model_dim*2
        self.model_dim, self.upscaled_dim, self.num_heads = model_dim, upscaled_dim, num_heads
        
        self.ln = nn.LayerNorm(model_dim, dtype=dtype)
        self.left_proj = nn.Linear(model_dim, upscaled_dim, bias=False, dtype=dtype) # upscale by projection factor of 2
        self.right_proj = nn.Linear(model_dim, upscaled_dim, bias=False, dtype=dtype) # upscale by projection factor of 2
        
        self.conv4 = CustomConv1D(dtype=dtype)
        self.lskip = nn.Linear(upscaled_dim,upscaled_dim,dtype=dtype)
        self.swish = Swish()
        
        self.query = nn.ModuleList([BlockDiagonalLinear(upscaled_dim, dtype=dtype) for _ in range(num_heads)])
        self.key = nn.ModuleList([BlockDiagonalLinear(upscaled_dim, scale=1/sqrt(upscaled_dim), dtype=dtype) for _ in range(num_heads)])
        self.value = nn.ModuleList([BlockDiagonalLinear(upscaled_dim, dtype=dtype) for _ in range(num_heads)])
        
        self.input_gate = nn.ModuleList([nn.Linear(upscaled_dim, 1, bias=False, dtype=dtype) for _ in range(num_heads)])
        self.forget_gate = nn.ModuleList([nn.Linear(upscaled_dim, 1, bias=False, dtype=dtype) for _ in range(num_heads)])
        self.output_gate = nn.ModuleList([nn.Linear(upscaled_dim, upscaled_dim, bias=False, dtype=dtype) for _ in range(num_heads)])
        
        self.group_norm =  nn.GroupNorm(num_groups=2, num_channels=upscaled_dim, dtype=dtype)
        
        self.down_proj = nn.Linear(num_heads*upscaled_dim, model_dim, bias=False, dtype=dtype)
        
    def _outer_product(self, x, y):
        
        assert x.shape==y.shape, "Need same shapes of x and y to calculate outer product."
        return torch.einsum('bi,bj->bij', x, y)
    
    def _batch_scalar_mat_mul(self, s, M):
        
        return s.view(-1,1,1) * M
    
    def _batch_matmul(self, M, v):
        
        return torch.einsum('bij,bj->bi', M, v)
    
    def _inner_abs_max(self, n, q, upper=1.0):
        
        dot_product = torch.einsum('bi,bi->b', n, q).unsqueeze(-1)
        abs_value = torch.abs(dot_product)
        max_value = torch.maximum(abs_value, torch.tensor(upper, dtype=q.dtype, device=q.device))
        
        return max_value
        
    def _step(self, x, x_conv_swish, C_n_tuple, hidx):
        
        assert x.shape[-1]==self.upscaled_dim, "Mismatch in the shape of x for stepping the mLSTM."
        C, n = C_n_tuple
        
        # calculate relevant quantities
        q, k, v = self.query[hidx](x_conv_swish), self.key[hidx](x_conv_swish), self.value[hidx](x)
        i_tilde, f_tilde, o_tilde = self.input_gate[hidx](x), self.forget_gate[hidx](x), self.output_gate[hidx](x)
        i, f, o = torch.exp(i_tilde), torch.sigmoid(f_tilde), torch.sigmoid(o_tilde)
        
        # update C
        C = self._batch_scalar_mat_mul(f,C) + self._batch_scalar_mat_mul(i,self._outer_product(v,k))
        # update n
        n = f*n + i*k
        
        # calculate output/hidden state h
        h_tilde = self._batch_matmul(C,q) / self._inner_abs_max(n,q)
        h = o*h_tilde
        
        return h, (C,n)
    
    def forward(self, x, C_n_tuple):
        
        C,n = C_n_tuple
        C,n = torch.split(C,C.shape[-1]//self.num_heads,dim=-1), torch.split(n,n.shape[-1]//self.num_heads,dim=-1)
        
        # Shape asserts
        assert x.shape[-1]==self.model_dim, "x shape error"
        for idx,(Ci,ni) in  enumerate(zip(C,n)):
            assert Ci.shape[1]==self.upscaled_dim and Ci.shape[2]==self.upscaled_dim, f"C shape error: idx {idx}"
            assert ni.shape[1]==self.upscaled_dim, f"n shape error: idx: {idx}"
            
        # forward
        
        # left
        x_ln = self.ln(x)
        x_left, x_right = self.left_proj(x_ln), self.right_proj(x_ln)
        x_conv_swish = self.swish(self.conv4(x_left))
        res = [self._step(x_left,x_conv_swish,(C[i],n[i]),i) for i in range(self.num_heads)]
        h = [a for a,_ in res]
        h_appended_normed = self.group_norm(torch.cat([hi[:,:,None] for hi in h], dim=-1))
        h_outs = torch.split(h_appended_normed,1,dim=-1)
        h_outs = [hi.squeeze(-1) for hi in h_outs]
        C_n_tuples = [b for _,b in res]
        C_tuples, n_tuples = [itm1 for (itm1,_) in C_n_tuples], [itm2 for (_,itm2) in C_n_tuples]
        ae = torch.cat(h_outs,dim=-1)
        be = self.lskip(x_left).repeat(1,self.num_heads)
        left_total = ae + be
        
        # right
        right_total = self.swish(x_right)
        
        # total, plus C and n
        total = left_total + right_total.repeat(1,self.num_heads)
        C,n = torch.cat(C_tuples,dim=-1), torch.cat(n_tuples,dim=-1)
        
        # return
        return x + self.down_proj(total), (C,n)
    
class mLSTM(nn.Module):
    
    def __init__(
        self,
        x_size: int,
        y_size: int,
        u_size: int,
        s_size: int,
        lookback: int,
        lookahead: int,
        model_dim: int = 64,
        num_heads: int = 4,
        dtype: torch.dtype = torch.float32
    ):
        
        super(mLSTM,self).__init__()
        
        in_features_past = x_size + y_size + u_size + s_size
        in_features_future = u_size
        out_features = y_size
        
        self.m_lstm_cell_past = mLSTMCell(model_dim=model_dim,num_heads=num_heads,dtype=dtype)
        self.m_lstm_cell_future = mLSTMCell(model_dim=model_dim,num_heads=num_heads,dtype=dtype)
        
        self.inp_proj_past = nn.Linear(in_features_past, model_dim, bias=False, dtype=dtype)
        self.inp_proj_future = nn.Linear(in_features_future, model_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(model_dim,out_features, dtype=dtype)
        
        self.lookback, self.lookahead = lookback, lookahead
        self.x_size, self.y_size, self.u_size, self.s_size = x_size, y_size, u_size, s_size
        self.model_dim, self.num_heads, self.dtype = model_dim, num_heads, dtype
        
    def _init_states(self, BS, device):
        
        C = torch.randn(BS, 2*self.model_dim, 2*self.model_dim*self.num_heads, dtype=self.dtype, device=device)
        n = torch.randn(BS, 2*self.model_dim*self.num_heads, dtype=self.dtype, device=device)
        
        return C,n
    
    def forward(self, x):
        
        # extract components
        y_past, x_past, u_past, s_past, u_future, _ = parse_inputs(x,self.lookback,self.lookahead,self.y_size,self.x_size,self.u_size,self.s_size)
        inp = torch.cat([y_past,x_past,u_past,s_past], dim=-1)
        BS = inp.shape[0]
        
        # generate states
        C,n = self._init_states(BS, inp.device)
        
        # roll out the past
        for t in range(self.lookback):
            _, (C,n) = self.m_lstm_cell_past(self.inp_proj_past(inp[:,t,:]),(C,n))
            
        # roll out the future
        outputs = []
        for t in range(self.lookahead):
            h, (C,n) = self.m_lstm_cell_future(self.inp_proj_future(u_future[:,t,:]),(C,n))
            outputs.append(self.out_proj(h)[:,None,:])
            
        # return
        return torch.cat(outputs,dim=1)