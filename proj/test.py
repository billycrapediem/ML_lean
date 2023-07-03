import numpy as np

data_state = np.load('state.npy')
data_vel = np.load('vel.npy')
data_Ft = np.load('Ft.npy')
#process the euler angle to normalize into [-1,1]
min_value = -np.pi
angle_range = 2 * np.pi
for i in range (3,6):
    data_state[i,:] = (data_state[i,:] - min_value) / (angle_range)
for i in range (0,3):
    data_Ft[i,:] = (data_Ft[i,:] - (-50)) / (100)
for i in range (3,6):
    print(data_Ft[i,:].min())
    data_Ft[i,:] = (data_Ft[i,:] - (-4)) / (8)
for i in range (3,6):
    print(data_Ft[i,:].min())
data = np.concatenate((data_state,data_vel),axis=0)