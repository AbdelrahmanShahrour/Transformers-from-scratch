import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class PositionWiseFeedForward(nn.Module):
    '''
    The PositionWiseFeedForward class extends PyTorch’s nn.Module 
    and implements a position-wise feed-forward network. The class 
    initializes with two linear transformation layers and a ReLU activation 
    function. The forward method applies these transformations and activation 
    function sequentially to compute the output. This process enables the model to 
    consider the position of input elements while making predictions.
    '''
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))