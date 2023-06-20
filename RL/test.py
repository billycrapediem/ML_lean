import torch
from torch.distributions import Categorical

probs = torch.tensor([0.8,0.5,0.1])
print(probs.shape)
pd = Categorical(probs=probs)

print(pd.sample())