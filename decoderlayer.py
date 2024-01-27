import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from multiheadattention import MultiHeadAttention
from positionwisefeedforward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        '''
        A Decoder layer consists of two Multi-Head Attention layers, a Position-wise Feed-Forward layer,
         and three Layer Normalization layers.

        The forward method computes the decoder layer output by performing the following steps:

        1. Calculate the masked self-attention output and add it to the input tensor, 
            followed by dropout and layer normalization.
        2. Compute the cross-attention output between the decoder and encoder outputs, 
            and add it to the normalized masked self-attention output, followed by 
            dropout and layer normalization.
        3. Calculate the position-wise feed-forward output and combine it with the 
            normalized cross-attention output, followed by dropout and layer normalization.
        4. Return the processed tensor.

        These operations enable the decoder to generate target sequences 
        based on the input and the encoder output.
        '''
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x