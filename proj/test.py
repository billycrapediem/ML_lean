import numpy as np


data_state = np.load('state.npy')
print(data_state[:1])
data_vel = np.load('vel.npy')
print(data_vel[:1])

data_Ft = np.load('Ft.npy')
print(data_Ft[:1])