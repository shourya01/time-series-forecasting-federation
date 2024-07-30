import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, sqrt
import numpy as np

# paper - https://proceedings.neurips.cc/paper_files/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html

class LogSparseMask():
    def __init__(self, B, L, local = 4, device="cpu"):
        mask = torch.zeros(B, 1, L, L, dtype=torch.bool)
        for i in range(L):
            mask[:, :, i, max(0, i - local):i + local + 1] = 1
            step = 1
            while i + step < L:
                mask[:, :, i, i + step] = 1
                step *= 2
            step = 1
            while i - step >= 0:
                mask[:, :, i, i - step] = 1
                step *= 2
        self._mask = ~mask.to(device)

    @property
    def mask(self):
        return self._mask

class LogSparseCausalMask():
    def __init__(self, B, L, local = 4, device="cpu"):
        mask = torch.zeros(B, 1, L, L, dtype=torch.bool)
        for i in range(L):
            mask[:, :, i, max(0, i - local):i + local + 1] = 1
            step = 1
            while i + step < L:
                mask[:, :, i, i + step] = 1
                step *= 2
            step = 1
            while i - step >= 0:
                mask[:, :, i, i - step] = 1
                step *= 2
        # Ensure the mask is strictly upper triangular
        mask = torch.triu(mask, diagonal=0)
        self._mask = ~mask.to(device)

    @property
    def mask(self):
        return self._mask

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", dtype=torch.float32):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, dtype=dtype)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, dtype=dtype)
        self.norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
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
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", dtype=torch.float32):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, dtype=dtype)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, dtype=dtype)
        self.norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, dtype=dtype)
        self.norm3 = nn.LayerNorm(d_model, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

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

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len=5000, dropout=0.1, dtype=torch.float32):
        super(DataEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(c_in, d_model, dtype=dtype)
        self.positional_embedding = PositionalEmbedding(d_model, max_len, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(x)
        return self.dropout(token_emb + pos_emb)
    
class FullAttentionEncoder(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttentionEncoder, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = LogSparseMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
class FullAttentionDecoder(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttentionDecoder, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = LogSparseCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, dtype=torch.float32):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
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

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class LogTransBase(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
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
        d_model = 64,
        factor = 3,
        n_heads = 4,
        d_ff = 512,
        d_layers = 2,
        e_layers = 2,
        activation = 'gelu',
        dropout = 0.1,
        dtype: torch.dtype = torch.float32
    ):
        
        super(LogTransBase, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, seq_len, dropout=dropout, dtype=dtype)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dec_len, dropout=dropout, dtype=dtype)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttentionEncoder(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads, dtype=dtype),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    dtype=dtype
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model, dtype=dtype)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttentionDecoder(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, dtype=dtype),
                    AttentionLayer(
                        FullAttentionEncoder(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, dtype=dtype),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    dtype=dtype
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model, dtype=dtype),
            projection=nn.Linear(d_model, c_out, bias=True, dtype=dtype)
        )

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
        
        
class LogTrans(nn.Module):
    
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
        
        super(LogTrans, self).__init__()
        
        enc_in = x_size + y_size + u_size + s_size
        dec_in = enc_in
        c_out = y_size
        seq_len, dec_len, out_len = lookback, lookback//2 + lookahead, lookahead
        
        # values to store
        self.dtype = dtype
        self.lookback, self.lookahead = lookback, lookahead
        self.x_size, self.y_size, self.u_size = x_size, y_size, u_size
        
        self.transformer = LogTransBase(
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
        
        y_past, x_past, u_past, s_past, u_future , _ = x
        enc_inp = torch.cat([y_past,x_past,u_past,s_past],dim=-1)
        dec_inp = torch.cat([
            enc_inp[:,-(self.lookback//2):,:],
            torch.zeros(enc_inp.shape[0],self.lookahead,enc_inp.shape[2], dtype=self.dtype, device=enc_inp.device)
        ], dim=1)
        
        # retcon u_future into the decoder inputs
        dec_inp[:,-self.lookahead:,self.x_size+self.y_size:self.x_size+self.y_size+self.u_size] = u_future
        
        return self.transformer(enc_inp,dec_inp)