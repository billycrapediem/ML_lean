import torch
import numpy as py
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim



## 建立
class QNet(nn.Module): 
    def __init__(self, n_hidden:int, n_dim: int, state_dim: int, action_dim: int,epls_rate):
        super().__init__()
        net_list = []
        net_list.extend([nn.Linear(2,n_dim),nn.ReLU()])
        for i in range (n_hidden):
            net_list.extend([nn.Linear(n_dim,n_dim)],nn.ReLU())
        net_list.append(nn.Linear(n_dim,1))
        self.net = nn.Sequential(*net_list)
        self.action_dim = action_dim
        self.epls_rate = epls_rate

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        return self.net(state)
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        if self.epls_rate < torch.rand(1):
            torch.argmax(self.net(state),dim=1,keepdim=True )
        else:
            action = torch.randint(self.action_dim, size = (state.shape[0],1))
        return action
    

