import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class lstm_net(nn.Module):
    def __init__(self,input_dim,mid_dim : 128,output_dim,hidden_layers):
        super(lstm_net,self).__init__()
        self.rnn = nn.LSTM(input_dim,mid_dim,hidden_layers)
        self.reg = nn.Sequentialj(
            nn.Linear(mid_dim,mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim,output_dim)
        )
    
    def forward(self, x):
        y = self.rnn(x)[0]