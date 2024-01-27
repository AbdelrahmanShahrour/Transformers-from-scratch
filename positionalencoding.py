import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class PositionalEncoding(nn.Module):
    '''
    Positional Encoding is used to inject the position information of each 
    token in the input sequence. It uses sine and cosine functions of different 
    frequencies to generate the positional encoding.

    The PositionalEncoding class initializes with input parameters d_model and max_seq_length, 
    creating a tensor to store positional encoding values. The class calculates sine and cosine 
    values for even and odd indices, respectively, based on the scaling factor div_term. 
    The forward method computes the positional encoding by adding the stored positional encoding 
    values to the input tensor, allowing the model to capture the position 
    information of the input sequence.
    '''
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]