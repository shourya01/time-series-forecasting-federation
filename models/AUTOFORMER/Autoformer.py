import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt, log

# paper - https://arxiv.org/pdf/2106.13008
# code (with many bugfixes for float16) from https://github.com/MAZiqing/FEDformer/blob/master/models/Autoformer.py

def next_power_of_two(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def pad_to_power_of_two(tensor):
    last_dim_size = tensor.size(-1)
    if (last_dim_size & (last_dim_size - 1)) != 0:
        padded_size = next_power_of_two(last_dim_size)
        pad_amount = padded_size - last_dim_size
        return torch.nn.functional.pad(tensor, (0, pad_amount)), padded_size
    return tensor, last_dim_size

def rfft_with_padding(input_tensor):
    padded_tensor, padded_size = pad_to_power_of_two(input_tensor)
    rfft_output = torch.fft.rfft(padded_tensor, dim=-1)
    return rfft_output[..., :input_tensor.size(-1)//2 + 1]

def irfft_with_padding(input_tensor):
    if input_tensor.dtype == torch.complex32:
        original_signal_length = (input_tensor.size(-1) - 1) * 2
        padded_size = next_power_of_two(original_signal_length)
        padded_tensor = torch.nn.functional.pad(input_tensor, (0, (padded_size // 2 + 1) - input_tensor.size(-1)))
        return torch.fft.irfft(padded_tensor, n=padded_size, dim=-1)
    else:
        return torch.fft.irfft(input_tensor, dim=-1)


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
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, activation="relu", dtype=torch.float32):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False, dtype=dtype)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False, dtype=dtype)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg, dtype)
            self.decomp2 = series_decomp_multi(moving_avg, dtype)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + new_x
        x, _ = self.decomp1(x)
        y = x
        y = self.activation(self.conv1(y.transpose(-1, 1)))
        y = self.conv2(y).transpose(-1, 1)
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
                 moving_avg=25, activation="relu", dtype=torch.float32):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False, dtype=dtype)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False, dtype=dtype)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg, dtype)
            self.decomp2 = series_decomp_multi(moving_avg, dtype)
            self.decomp3 = series_decomp_multi(moving_avg, dtype)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False, dtype=dtype)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0]

        x, trend1 = self.decomp1(x)
        x = x + self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0]

        x, trend2 = self.decomp2(x)
        y = x
        y = self.activation(self.conv1(y.transpose(-1, 1)))
        y = self.conv2(y).transpose(-1, 1)
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


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

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

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
    
class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.agg = None
        # self.use_wavelet = configs.wavelet

    # @decor_time
    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg  # size=[B, H, d, S]

    def time_delay_agg_inference(self, values, corr):
        """     
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

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

        # period-based dependencies
        q_fft = rfft_with_padding(queries.permute(0, 2, 3, 1).contiguous())
        k_fft = rfft_with_padding(keys.permute(0, 2, 3, 1).contiguous())
        res = q_fft * torch.conj(k_fft)
        corr = irfft_with_padding(res)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


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
        
        # hacky fix to dtype error when torch.float16, to be investigated indepth later
        target_dtype_for_out = self.out_projection.weight.dtype
        
        return self.out_projection(out.to(target_dtype_for_out)), attn
    
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

class AutoformerBase(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
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
        factor = 3,
        n_heads = 4,
        d_ff = 512,
        d_layers = 2,
        e_layers = 2,
        activation = 'gelu',
        dtype: torch.dtype = torch.float32
    ):
        super(AutoformerBase, self).__init__()
        self.seq_len = seq_len
        self.dec_len = dec_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.dtype = dtype

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp(kernel_size[0])
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = TokenEmbedding(enc_in, d_model, dtype=dtype)
        self.dec_embedding = TokenEmbedding(dec_in, d_model, dtype=dtype)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor,
                                        output_attention=output_attention),
                        d_model, n_heads, dtype=dtype),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    activation=activation,
                    dtype = dtype
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model, dtype=dtype)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor,
                                        output_attention=False),
                        d_model, n_heads, dtype = dtype),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor,
                                        output_attention=False),
                        d_model, n_heads, dtype=dtype),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    activation=activation,
                    dtype = dtype
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model, dtype=dtype),
            projection=nn.Linear(d_model, c_out, bias=True, dtype=dtype)
        )

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], dtype=self.dtype, device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.dec_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.dec_len:, :], zeros], dim=1)
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
            return dec_out[:, -self.pred_len:, :]
        
        
class Autoformer(nn.Module):
    
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
        
        super(Autoformer, self).__init__()
        
        enc_in = x_size + y_size + u_size + s_size
        dec_in = enc_in
        c_out = y_size
        seq_len, dec_len, out_len = lookback, lookback//2 + lookahead, lookahead
        
        # values to store
        self.dtype = dtype
        self.lookback, self.lookahead = lookback, lookahead
        self.x_size, self.y_size, self.u_size = x_size, y_size, u_size
        
        self.autoformer = AutoformerBase(
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
        
        y_past, x_past, u_past, s_past, u_future , _ = x
        enc_inp = torch.cat([y_past,x_past,u_past,s_past],dim=-1)
        dec_inp = torch.cat([
            enc_inp[:,-(self.lookback//2):,:],
            torch.zeros(enc_inp.shape[0],self.lookahead,enc_inp.shape[2], dtype=self.dtype, device=enc_inp.device)
        ], dim=1)
        
        # retcon u_future into the decoder inputs
        dec_inp[:,-self.lookahead:,self.x_size+self.y_size:self.x_size+self.y_size+self.u_size] = u_future
        
        return self.autoformer(enc_inp,dec_inp)[:,:,:self.y_size] # only output the y's