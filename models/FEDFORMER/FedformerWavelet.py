import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
from functools import partial
from typing import List, Tuple
from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt

# paper - https://arxiv.org/pdf/2201.12740
# code adapted from - https://github.com/MAZiqing/FEDformer/blob/master/models/FEDformer.py

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

def legendreDer(k, x):
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k-1,-1,-2):
        out += _legendre(i, x)
    return out


def phi_(phi_c, x, lb = 0, ub = 1):
    mask = np.logical_or(x<lb, x>ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)


def get_phi_psi(k, base):
    
    x = Symbol('x')
    phi_coeff = np.zeros((k,k))
    phi_2x_coeff = np.zeros((k,k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2*x-1), x).all_coeffs()
            phi_coeff[ki,:ki+1] = np.flip(np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4*x-1), x).all_coeffs()
            phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                a = phi_2x_coeff[ki,:ki+1]
                b = phi_coeff[i, :i+1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_)<1e-8] = 0
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]
            for j in range(ki):
                a = phi_2x_coeff[ki,:ki+1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_)<1e-8] = 0
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            a = psi1_coeff[ki,:]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_)<1e-8] = 0
            norm1 = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki,:]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_)<1e-8] = 0
            norm2 = (prod_ * 1/(np.arange(len(prod_))+1) * (1-np.power(0.5, 1+np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i,:])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i,:])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i,:])) for i in range(k)]
    
    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki,:ki+1] = np.sqrt(2/np.pi)
                phi_2x_coeff[ki,:ki+1] = np.sqrt(2/np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2*x-1), x).all_coeffs()
                phi_coeff[ki,:ki+1] = np.flip(2/np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4*x-1), x).all_coeffs()
                phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                
        phi = [partial(phi_, phi_coeff[i,:]) for i in range(k)]
        
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2)* phi[ki](2*x_m)).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2*x_m)).sum()        
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5, ub = 1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5+1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5+1e-16, ub = 1)
        
    return phi, psi1, psi2

def get_filter(base, k):
    
    def psi(psi1, psi2, i, inp):
        mask = (inp<=0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1-mask)
    
    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')
    
    x = Symbol('x')
    H0 = np.zeros((k,k))
    H1 = np.zeros((k,k))
    G0 = np.zeros((k,k))
    G1 = np.zeros((k,k))
    PHI0 = np.zeros((k,k))
    PHI1 = np.zeros((k,k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1/k/legendreDer(k,2*x_m-1)/eval_legendre(k-1,2*x_m-1)
        
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()
                
        PHI0 = np.eye(k)
        PHI1 = np.eye(k)
                
    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()

                PHI0[ki, kpi] = (wm * phi[ki](2*x_m) * phi[kpi](2*x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2*x_m-1) * phi[kpi](2*x_m-1)).sum() * 2
                
        PHI0[np.abs(PHI0)<1e-8] = 0
        PHI1[np.abs(PHI1)<1e-8] = 0

    H0[np.abs(H0)<1e-8] = 0
    H1[np.abs(H1)<1e-8] = 0
    G0[np.abs(G0)<1e-8] = 0
    G1[np.abs(G1)<1e-8] = 0
        
    return H0, H1, G0, G1, PHI0, PHI1

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels, dtype=torch.float32):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels, dtype=dtype)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, dtype=torch.float32):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size), dtype=dtype)

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu", dtype=torch.float32):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False, dtype=dtype)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False, dtype=dtype)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg, dtype=dtype)
            self.decomp2 = series_decomp_multi(moving_avg, dtype=dtype)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu", dtype=torch.float32):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False, dtype=dtype)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False, dtype=dtype)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg, dtype=dtype)
            self.decomp2 = series_decomp_multi(moving_avg, dtype=dtype)
            self.decomp3 = series_decomp_multi(moving_avg, dtype=dtype)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False, dtype=dtype)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
    
class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None, dtype=torch.float32):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, dtype=dtype)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads, dtype=dtype)
        self.value_projection = nn.Linear(d_model, d_values * n_heads, dtype=dtype)
        self.out_projection = nn.Linear(d_values * n_heads, d_model, dtype=dtype)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn
    
class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dtype=torch.float32):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular', dtype=dtype)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, dtype=torch.float32):
        super(DataEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(c_in, d_model, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        return self.dropout(token_emb)
    
class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 dtype=torch.float32,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()
        
        complexMatch = {
            torch.float32:torch.complex64,
            torch.float64:torch.complex128
        }
        self.cdtype = complexMatch[dtype]

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=self.cdtype))
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        # l = N//2+1
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=self.cdtype)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x
    
class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k=3, alpha=64,
                 L=0, c=1,
                 base='legendre',
                 initializer=None,
                 dtype=torch.float32,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        self.dtype = dtype
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.A = sparseKernelFT1d(k, alpha, c, dtype=dtype)
        self.B = sparseKernelFT1d(k, alpha, c, dtype=dtype)
        self.C = sparseKernelFT1d(k, alpha, c, dtype=dtype)

        self.T0 = nn.Linear(k, k, dtype=dtype)

        self.register_buffer('ec_s', torch.tensor(
            np.concatenate((H0.T, H1.T), axis=0), dtype=dtype))
        self.register_buffer('ec_d', torch.tensor(
            np.concatenate((G0.T, G1.T), axis=0), dtype=dtype))

        self.register_buffer('rc_e', torch.tensor(
            np.concatenate((H0r, G0r), axis=0), dtype=dtype))
        self.register_buffer('rc_o', torch.tensor(
            np.concatenate((H1r, G1r), axis=0), dtype=dtype))

    def forward(self, x):
        
        B, N, c, k = x.shape  # (B, N, k)
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0:nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        #         decompose
        for i in range(ns - self.L):
            # print('x shape',x.shape)
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform
        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]

        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k, dtype=self.dtype,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x
    
class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(self, ich=1, k=8, alpha=16, c=128,
                 nCZ=1, L=0, base='legendre', attention_dropout=0.1, dtype=torch.float32):
        super(MultiWaveletTransform, self).__init__()
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k, dtype=dtype)
        self.Lk1 = nn.Linear(c * k, ich, dtype=dtype)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base, dtype=dtype) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        values = values.view(B, L, -1)

        V = self.Lk0(values).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)
        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return (V.contiguous(), None)
    
class FourierCrossAttentionW(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh',
                 mode_select_method='random',dtype=torch.float32):
        super(FourierCrossAttentionW, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation
        
        complexMatch = {
            torch.float32:torch.complex64,
            torch.float64:torch.complex128
        }
        self.cdtype = complexMatch[dtype]

    def forward(self, q, k, v, mask):
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=self.cdtype)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=self.cdtype)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=self.cdtype)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
            
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        return (out, None)

class MultiWaveletCross(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64,
                 k=8, ich=512,
                 L=0,
                 base='legendre',
                 mode_select_method='random',
                 initializer=None, activation='tanh',
                 dtype=torch.float32,
                 **kwargs):
        super(MultiWaveletCross, self).__init__()

        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method,dtype=dtype)
        self.attn2 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method,dtype=dtype)
        self.attn3 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method,dtype=dtype)
        self.attn4 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method,dtype=dtype)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.tensor(
            np.concatenate((H0.T, H1.T), axis=0),dtype=dtype))
        self.register_buffer('ec_d', torch.tensor(
            np.concatenate((G0.T, G1.T), axis=0),dtype=dtype))

        self.register_buffer('rc_e', torch.tensor(
            np.concatenate((H0r, G0r), axis=0),dtype=dtype))
        self.register_buffer('rc_o', torch.tensor(
            np.concatenate((H1r, G1r), axis=0),dtype=dtype))

        self.Lk = nn.Linear(ich, c * k, dtype=dtype)
        self.Lq = nn.Linear(ich, c * k, dtype=dtype)
        self.Lv = nn.Linear(ich, c * k, dtype=dtype)
        self.out = nn.Linear(c * k, ich, dtype=dtype)
        self.modes1 = modes
        self.dtype=dtype

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        dtype=self.dtype,device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x

class FedformerWaveletBase(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        dec_len,
        pred_len,
        output_attention = False,
        moving_avg = [4,8],
        L = 1,
        base = 'legendre',
        d_model = 64,
        modes = 32,
        mode_select = 'random',
        cross_activation = 'tanh',
        n_heads = 4,
        d_ff = 512,
        d_layers = 2,
        e_layers = 2,
        activation = 'gelu',
        dropout = 0.1,
        dtype: torch.dtype = torch.float32
    ):
        
        super(FedformerWaveletBase, self).__init__()
        
        self.mode_select = mode_select
        self.modes = modes
        self.seq_len = seq_len
        self.label_len = dec_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.dtype = dtype

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size, dtype=dtype)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout=dropout, dtype=dtype)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout=dropout, dtype=dtype)

        encoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base, dtype=dtype)
        decoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base, dtype=dtype)
        decoder_cross_att = MultiWaveletCross(in_channels=d_model,
                                                out_channels=d_model,
                                                seq_len_q=self.seq_len // 2 + self.pred_len,
                                                seq_len_kv=self.seq_len,
                                                modes=modes,
                                                ich=d_model,
                                                base=base,
                                                activation=cross_activation,
                                                dtype=dtype)
        
        # Encoder
        enc_modes = int(min(modes, seq_len//2))
        dec_modes = int(min(modes, (seq_len//2+pred_len)//2))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        d_model, n_heads, dtype=dtype),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                    dtype=dtype
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model, dtype=dtype)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        d_model, n_heads, dtype=dtype),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        d_model, n_heads, dtype=dtype),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                    dtype=dtype
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model, dtype=dtype),
            projection=nn.Linear(d_model, c_out, bias=True, dtype=dtype)
        )

    def forward(self, x_enc,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
class FedformerWavelet(nn.Module):
    
    def __init__(
        self,
        x_size: int,
        y_size: int,
        u_size: int,
        s_size: int,
        lookback: int,
        lookahead: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 512,
        activation: str = 'gelu',
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32
    ):
        
        super(FedformerWavelet, self).__init__()
        
        enc_in = x_size + y_size + u_size + s_size
        dec_in = enc_in
        c_out = y_size
        seq_len, dec_len, out_len = lookback, lookback//2 + lookahead, lookahead
        
        # values to store
        self.dtype = dtype
        self.lookback, self.lookahead = lookback, lookahead
        self.x_size, self.y_size, self.u_size, self.s_size = x_size, y_size, u_size, s_size
        
        self.autoformer = FedformerWaveletBase(
            enc_in = enc_in,
            dec_in = dec_in,
            c_out = c_out,
            seq_len = seq_len,
            dec_len = dec_len,
            pred_len = out_len,
            d_model = d_model,
            n_heads = n_heads,
            e_layers = e_layers,
            d_layers = d_layers,
            d_ff = d_ff,
            activation = activation,
            dropout = dropout,
            dtype = dtype
        )
                
    def forward(self, x):
        
        y_past, x_past, u_past, s_past, _ , _ = parse_inputs(x,self.lookback,self.lookahead,self.y_size,self.x_size,self.u_size,self.s_size)
        enc_inp = torch.cat([y_past,x_past,u_past,s_past],dim=-1)
        
        return self.autoformer(enc_inp)[:,:,:self.y_size] # only output the y's