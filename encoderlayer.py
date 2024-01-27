import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from multiheadattention import MultiHeadAttention
from positionwisefeedforward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    '''
    An Encoder layer consists of a Multi-Head Attention layer,
    a Position-wise Feed-Forward layer, and two Layer Normalization layers.

    The EncoderLayer class initializes with input parameters and components, 
    including a MultiHeadAttention module, a PositionWiseFeedForward module, two 
    layer normalization modules, and a dropout layer. The forward methods computes 
    the encoder layer output by applying self-attention, adding the attention 
    output to the input tensor, and normalizing the result. Then, it 
    computes the position-wise feed-forward output, combines it with the 
    normalized self-attention output, and normalizes the final result 
    before returning the processed tensor.
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x