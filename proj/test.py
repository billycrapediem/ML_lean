import numpy as np
data_state = np.load('state.npy')
data_vel = np.load('vel.npy')
data_Ft = np.load('Ft.npy')

for i in range(6):
    print(data_Ft[i,:].shape,data_Ft[i,:].min())