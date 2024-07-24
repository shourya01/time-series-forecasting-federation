import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# paper - https://arxiv.org/pdf/2201.12740
# code adapted from - https://github.com/MAZiqing/FEDformer/blob/master/models/FEDformer.py

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

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
    
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, n_heads, head_size,
                 modes=0, mode_select_method='random', dtype=torch.float32,):
        super(FourierBlock, self).__init__()
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        
        # matching complex type to corresponding float type
        complexMatch = {
            torch.float32:torch.complex64,
            torch.float64:torch.complex128
        }
        self.cdtype = complexMatch[dtype]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(n_heads, n_heads, head_size, len(self.index), dtype=self.cdtype))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        result = torch.einsum("bix,iox->box", input, weights)
        return result

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=self.cdtype)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)
    
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, n_heads, head_size,
                 modes=64, mode_select_method='random',
                 activation='tanh', policy=0, dtype=torch.float32):
        super(FourierCrossAttention, self).__init__()
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        
        # matching complex type to corresponding float type
        complexMatch = {
            torch.float32:torch.complex64,
            torch.float64:torch.complex128
        }
        self.cdtype = complexMatch[dtype]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(n_heads, head_size, head_size, len(self.index_q), dtype=self.cdtype))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=self.cdtype)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=self.cdtype)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} activation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=self.cdtype)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)

class FedformerFourierBase(nn.Module):
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
        d_model = 64,
        modes = 32,
        mode_select = 'random',
        n_heads = 4,
        d_ff = 512,
        d_layers = 2,
        e_layers = 2,
        activation = 'gelu',
        dropout = 0.1,
        dtype: torch.dtype = torch.float32
    ):
        
        super(FedformerFourierBase, self).__init__()
        
        self.mode_select = mode_select
        self.modes = modes
        self.seq_len = seq_len
        self.label_len = dec_len
        self.pred_len = pred_len
        self.output_attention = output_attention

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

        encoder_self_att = FourierBlock(in_channels=d_model,
                                        out_channels=d_model,
                                        seq_len=self.seq_len,
                                        n_heads=n_heads,
                                        head_size=(d_model // n_heads),
                                        modes=modes,
                                        mode_select_method=mode_select,
                                        dtype=dtype)
        decoder_self_att = FourierBlock(in_channels=d_model,
                                        out_channels=d_model,
                                        seq_len=self.seq_len//2+self.pred_len,
                                        n_heads=n_heads,
                                        head_size=(d_model // n_heads),
                                        modes=modes,
                                        mode_select_method=mode_select,
                                        dtype=dtype)
        decoder_cross_att = FourierCrossAttention(in_channels=d_model,
                                                    out_channels=d_model,
                                                    seq_len_q=self.seq_len//2+self.pred_len,
                                                    seq_len_kv=self.seq_len,
                                                    n_heads=n_heads,
                                                    head_size=(d_model // n_heads),
                                                    modes=modes,
                                                    mode_select_method=mode_select,
                                                    dtype=dtype)
        # Encoder

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
        
class FedformerFourier(nn.Module):
    
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
        dtype: torch.dtype = torch.float32
    ):
        
        super(FedformerFourier, self).__init__()
        
        enc_in = x_size + y_size + u_size + s_size
        dec_in = enc_in
        c_out = y_size
        seq_len, dec_len, out_len = lookback, lookback//2 + lookahead, lookahead
        
        # values to store
        self.dtype = dtype
        self.lookback, self.lookahead = lookback, lookahead
        self.x_size, self.y_size, self.u_size = x_size, y_size, u_size
        
        self.autoformer = FedformerFourierBase(
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
            dtype = dtype
        )
                
    def forward(self, x):
        
        y_past, x_past, u_past, s_past, _ , _ = x
        enc_inp = torch.cat([y_past,x_past,u_past,s_past],dim=-1)
        
        return self.autoformer(enc_inp)[:,:,:self.y_size] # only output the y's