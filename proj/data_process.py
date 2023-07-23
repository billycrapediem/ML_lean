import numpy as np
def data_preprocess():
    data_state = np.load('state.npy')
    print(data_state)
    data_vel = np.load('vel.npy')
    data_Ft = np.load('Ft.npy')
    #process the euler angle to normalize into [-1,1]
    min_value = -np.pi
    angle_range = 2 * np.pi
    # normalize the angle 
    for i in range (3,6):
        data_state[i,:] = (data_state[i,:] - min_value) / (angle_range)
    for i in range(6,8):
        data_state[i,:] = (data_state[i,:] + 20) / (40)
    '''
    try to normalize the velocity
    1: 10 -10
    2: 10 -106
    3: 10 -10
    4: 10 -10
    '''
    for i in range(0,8):
        data_vel[i,:] = (data_vel[i,:] + 10) / 20
    
    '''
    try to normalize the output with 
    1 :40 -40
    2 40 -40
    3 40 -40
    4 20 -20
    5 20 -20
    6 10 -10
    '''

    '''
    for i in range(0,3):
        data_Ft[i,:] = (data_Ft[i,:] + 40) / 80
    for i in range(3,6):
        data_Ft[i,:] = (data_Ft[i,:] + 20) / 40
    '''
    data = np.concatenate((data_state,data_vel),axis=0)
    data = np.transpose(data)
    data_Ft = np.transpose(data_Ft)
    #return data,data_Ft
    data = np.concatenate((data,data_Ft),axis=1)
    #np.random.shuffle(data)
    np.save('lstm_process_data.npy',data) 

if __name__ == '__main__':
    data_preprocess()