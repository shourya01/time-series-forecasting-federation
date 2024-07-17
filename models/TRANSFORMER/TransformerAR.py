import torch
import torch.nn as nn
import math

# THEORETICAL NOTE:
# The cross-attention mechanism in the Transformer decoder deals with different and arbitrary input_seq_len and target_seq_len by computing attention 
# weights independently for each target token with respect to all input tokens. The attention mechanism is fundamentally a weighted sum of values 
# (from the encoder's output) where the weights (attention scores) are calculated using a query (from the decoder's current token) against all keys 
# (from the encoder's output).
# The runtime of transformer is O(input_seq_len*target_seq_len)

class TransformerAR(nn.Module):
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
        num_encoder_layers: int = 2, 
        num_decoder_layers: int = 2,
        dtype: torch.dtype = torch.float32
        ):
        
        super(TransformerAR, self).__init__()
        
        # Collect dimensions
        input_dim = y_size + x_size + u_size + s_size
        target_dim = y_size + u_size
        input_seq_len = lookback
        target_seq_len = lookahead + 1 # for zero starting token
        self.model_dim = model_dim
        self.target_seq_len = target_seq_len
        
        # Save data type
        self.dtype = dtype
        
        # Make relevant layers
        self.input_projection = nn.Linear(input_dim, model_dim, dtype=dtype)
        self.target_projection = nn.Linear(target_dim, model_dim, dtype=dtype)
        self.output_projection = nn.Linear(model_dim, target_dim, dtype=dtype)
        self.register_buffer('positional_encoding', self.create_sinusoidal_encoding(max(input_seq_len, target_seq_len), model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dtype=dtype)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dtype=dtype)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def create_sinusoidal_encoding(self, seq_len, model_dim):
        position = torch.arange(seq_len, dtype=self.dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype=self.dtype) * -(math.log(10000.0) / model_dim))
        sinusoidal_encoding = torch.zeros(seq_len, model_dim, dtype=self.dtype)
        sinusoidal_encoding[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_encoding[:, 1::2] = torch.cos(position * div_term)
        return sinusoidal_encoding.unsqueeze(0)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=self.dtype, device=self.positional_encoding.device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x, mode='train'):
        
        # extract data
        y_past, x_past, u_past, s_past, u_future, y_target = x
        
        # create relevant tensors
        src = torch.cat([y_past,x_past,u_past,s_past], dim=-1)
        
        src = self.input_projection(src)
        src += self.positional_encoding[:, :src.size(1), :]
        memory = self.transformer_encoder(src)

        if mode == 'train':
            # For training, use teacher forcing
            # Create relevant target
            tgt0 = torch.cat([y_target,u_future],dim=-1)
            tgt_pad = torch.zeros(tgt0.shape[0],1,tgt0.shape[2], dtype=self.dtype)
            tgt = torch.cat([tgt_pad,tgt0],dim=1)
            # now use this target
            tgt = self.target_projection(tgt)
            tgt += self.positional_encoding[:, :tgt.size(1), :]
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        else:
            if mode == 'test':
                # For testing, use autoregression
                output = torch.zeros((src.size(0), self.target_seq_len, self.model_dim), dtype=self.dtype, device=src.device)
                for i in range(self.target_seq_len):
                    tgt = output[:, :i+1, :]
                    tgt += self.positional_encoding[:, :i+1, :]
                    tgt_mask = self.generate_square_subsequent_mask(i+1)
                    out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
                    if i != self.target_seq_len-1:
                        candidate_output = out[:, [-1], :]
                        candidate_output_actual_value = self.output_projection(candidate_output)
                        candidate_output_actual_value[:,:,y_target.shape[2]:] = u_future[:,[i],:] # 'correct' the u values in predicted output
                        out_new = self.target_projection(candidate_output_actual_value)
                        output[:, i, :] = out_new[:, -1, :]
            else:
                raise ValueError('In forward() of TransformerAR, thew keywork <mode> received an unrecognized keyword; must be <train> or <test>.')

        return self.output_projection(output[:,:-1,:])[:,:,:y_target.shape[2]]