import torch
from ReLu_net import neuro_net
network = neuro_net(16,6)
network.load_state_dict(torch.load('Relu_neuro_net.pt'))